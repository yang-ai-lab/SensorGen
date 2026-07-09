"""SensorGen inference entry point.

A single-process (single-GPU or CPU) runner for the two released SensorGen
checkpoints. It builds the task-specific model, loads the checkpoint weights,
runs ODE sampling, and writes the outputs.

Examples
--------
    python -m sensorgen.inference --task text2ecg \
        --checkpoint ./ckpts/text2ecg.pt --config configs/text2ecg.yaml \
        --output_dir outputs/text2ecg

    python -m sensorgen.inference --task bp_translation \
        --checkpoint ./ckpts/bp_translation.pt --config configs/bp_translation.yaml \
        --output_dir outputs/bp_translation

If ``--checkpoint`` is omitted the checkpoint is downloaded from Hugging Face
(``yang-ai-lab/SensorGen``). Data paths come from the YAML config
(``--config``; defaults to ``configs/<task>.yaml``).
"""

import argparse
import os
import random
from copy import deepcopy

import numpy as np
import torch
import yaml

from sensorgen import TASKS
from sensorgen._backbone import REPO_ROOT, ensure_backbone_on_path
from sensorgen.checkpoint import resolve_checkpoint
from sensorgen.output import save_results

_CONFIG_DIR = os.path.join(REPO_ROOT, "configs")

# Internal architecture identifier used by the bundled model registry. Kept out
# of the public configs/docs; both released checkpoints share this architecture.
_BACKBONE_ARCH = "B/1d"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(task, config_path=None):
    path = config_path or os.path.join(_CONFIG_DIR, f"{task}.yaml")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# --------------------------------------------------------------------------- #
# Model build + checkpoint load
# --------------------------------------------------------------------------- #
def build_model(cfg, device):
    ensure_backbone_on_path()
    from models import MODELS  # internal model registry

    task_type = cfg["task_type"]
    series_length = cfg["series_length"]
    in_channels = cfg["in_channels"]
    patch_size = cfg["patch_size"]
    num_classes = cfg.get("num_classes", 1000)
    class_dropout_prob = cfg.get("class_dropout_prob", 0.1)

    if task_type == "generation":
        # Text conditioning path (report -> pooled AdaLN + token cross-attention).
        extra = dict(
            use_temporal_y_embedder=False,
            use_temporal_c2_embedder=False,
            use_resnet_y_embedder=False,
            use_label_y_embedder=False,
            y_encoder_config=None,
            c2_encoder_config=None,
        )
    elif task_type == "cross_channel":
        # Sparse vector -> pooled AdaLN condition; dense waveform -> cross-attention.
        extra = dict(
            use_temporal_y_embedder=True,
            use_temporal_c2_embedder=True,
            use_resnet_y_embedder=False,
            use_label_y_embedder=False,
            y_encoder_config={
                "input_len": series_length,
                "in_channels": cfg["y_cond_channels"],
                "num_patches": series_length // patch_size,
                "pool_type": cfg.get("y_pool_type", "mean"),
            },
            c2_encoder_config={
                "input_len": series_length,
                "in_channels": cfg["c2_in_channels"],
                "patch_size": patch_size,
                "depth": cfg.get("c2_depth", 2),
            },
        )
    else:
        raise ValueError(f"Unsupported task_type '{task_type}'.")

    model = MODELS[_BACKBONE_ARCH](
        input_size=series_length,
        num_classes=num_classes,
        class_dropout_prob=class_dropout_prob,
        in_channels=in_channels,
        input_channels=None,
        patch_size=patch_size,
        **extra,
    )
    return model.to(device)


def load_weights(model, ckpt_path, device):
    """Load the sampling (EMA) weights from a checkpoint into a fresh copy."""
    net = deepcopy(model).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "ema" in state:
        net.load_state_dict(state["ema"])
    elif isinstance(state, dict) and "model" in state:
        net.load_state_dict(state["model"])
    else:
        net.load_state_dict(state)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def build_loader(task, cfg, batch_size, num_workers, split):
    if task == "text2ecg":
        from sensorgen.data.mimic_ecg import build_mimic_ecg_loader

        return build_mimic_ecg_loader(
            h5_dir=cfg["h5_dir"],
            csv_path=cfg["csv_path"],
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            target_sr=cfg["target_sr"],
            seg_len=cfg["seg_len"],
            text_col=cfg.get("text_col", "total_report"),
        )
    from sensorgen.data.vitaldb_bp import build_vitaldb_bp_loader

    return build_vitaldb_bp_loader(
        h5_path=cfg["h5_path"],
        split=split,
        target_sr=cfg["target_sr"],
        x_signal=cfg["x_signals"][0],
        c2_signal=cfg["c2_signals"][0],
        c1_signal=cfg["y_signals"][0],
        batch_size=batch_size,
        num_workers=num_workers,
    )


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def _make_sampler(cfg):
    ensure_backbone_on_path()
    from transport import Sampler, create_transport  # internal sampler

    transport = create_transport(
        cfg.get("path_type", "Linear"),
        cfg.get("prediction", "velocity"),
        cfg.get("loss_weight"),
        cfg.get("train_eps"),
        cfg.get("sample_eps"),
    )
    return Sampler(transport).sample_ode()


def _stack(chunks):
    if not chunks:
        return None
    return torch.cat(chunks, dim=0).numpy()


def run_inference(task, cfg, model, loader, args, device):
    sample_fn = _make_sampler(cfg)
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else float(cfg.get("cfg_scale", 1.0))
    use_cfg = cfg_scale > 1.0
    model_fn = model.forward_with_cfg if use_cfg else model.forward
    in_channels = cfg["in_channels"]
    series_length = cfg["series_length"]
    target_n = args.num_samples

    generated, real, contexts, nibp = [], [], [], []
    prompts = []
    collected = 0

    if task == "text2ecg" and cfg.get("use_prompting", False):
        from prompt_generator import generate_random_formats
    else:
        generate_random_formats = None

    for batch in loader:
        if collected >= target_n:
            break

        if task == "text2ecg":
            signals, texts = batch
            signals = signals.to(device)
            reports = list(texts)
            if generate_random_formats is not None:
                reports = generate_random_formats(reports)
            bs = signals.size(0)
            zs = torch.randn(bs, in_channels, series_length, device=device)
            ys = list(reports)
            if use_cfg:
                ys = ys + [""] * bs
                zs = torch.cat([zs, zs], dim=0)
                kwargs = dict(y=ys, c2=None, cfg_scale=cfg_scale)
            else:
                kwargs = dict(y=ys, c2=None)
            with torch.no_grad():
                out = sample_fn(zs, model_fn, **kwargs)[-1]
            if use_cfg:
                out, _ = out.chunk(2, dim=0)
            generated.append(out.detach().cpu())
            real.append(signals.detach().cpu())
            prompts.extend(reports)
            collected += bs
        else:  # bp_translation
            x = batch["x"].to(device)
            c2 = batch["c2"].to(device)
            y = batch["y"]
            if y is not None:
                y = y.to(device)
            bs = x.size(0)
            zs = torch.randn(bs, in_channels, series_length, device=device)
            with torch.no_grad():
                out = sample_fn(zs, model_fn, y=y, c2=c2)[-1]
            generated.append(out.detach().cpu())
            real.append(x.detach().cpu())
            contexts.append(c2.detach().cpu())
            if y is not None:
                nibp.append(y.detach().cpu())
            collected += bs

    def _trunc(arr):
        return None if arr is None else arr[:target_n]

    return {
        "generated": _trunc(_stack(generated)),
        "real": _trunc(_stack(real)),
        "contexts": _trunc(_stack(contexts)),
        "nibp": _trunc(_stack(nibp)),
        "prompts": prompts[:target_n] if prompts else None,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        description="SensorGen inference (text2ecg / bp_translation).",
    )
    p.add_argument("--task", required=True, choices=list(TASKS),
                   help="Which released task to run.")
    p.add_argument("--config", default=None,
                   help="YAML config path (default: configs/<task>.yaml).")
    p.add_argument("--checkpoint", "--ckpt", dest="checkpoint", default=None,
                   help="Local checkpoint path. If omitted, download from Hugging Face "
                        "(yang-ai-lab/SensorGen).")
    p.add_argument("--output_dir", "--output-dir", dest="output_dir",
                   default="./outputs",
                   help="Directory to write generated samples + manifest.json.")
    p.add_argument("--num-samples", "--num_samples", dest="num_samples", type=int, default=64,
                   help="Number of samples to generate.")
    p.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=32)
    p.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=4)
    p.add_argument("--split", default=None,
                   help="Dataset split to sample from (default: config 'split' or 'test').")
    p.add_argument("--cfg-scale", "--cfg_scale", dest="cfg_scale", type=float, default=None,
                   help="Classifier-free guidance scale (>1 enables CFG; text2ecg only). "
                        "Default: config value.")
    p.add_argument("--device", default=None,
                   help="Torch device (default: cuda if available else cpu).")
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config(args.task, args.config)
    split = args.split or cfg.get("split", "test")

    ckpt_path = resolve_checkpoint(args.task, ckpt=args.checkpoint)
    print(f"[sensorgen] task={args.task} device={device} checkpoint={ckpt_path}")

    model = build_model(cfg, device)
    model = load_weights(model, ckpt_path, device)

    loader = build_loader(args.task, cfg, args.batch_size, args.num_workers, split)
    results = run_inference(args.task, cfg, model, loader, args, device)

    if results["generated"] is None:
        raise RuntimeError("No samples were produced; check the dataset paths in the config.")

    manifest = save_results(
        args.output_dir, args.task, results, checkpoint=os.path.basename(ckpt_path), config=cfg
    )
    print(f"[sensorgen] wrote {manifest['num_samples']} samples to {args.output_dir}")
    print(f"[sensorgen] files: {manifest['files']}")


if __name__ == "__main__":
    main()
