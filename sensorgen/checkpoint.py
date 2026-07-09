"""Resolve SensorGen checkpoints from a local path or the Hugging Face Hub.

The two released checkpoints live in the Hub repo ``yang-ai-lab/SensorGen``::

    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id="yang-ai-lab/SensorGen", filename="text2ecg.pt")
    hf_hub_download(repo_id="yang-ai-lab/SensorGen", filename="bp_translation.pt")
"""

import os

HF_REPO_ID = "yang-ai-lab/SensorGen"

# Task -> canonical checkpoint filename on the Hub.
CHECKPOINT_FILES = {
    "text2ecg": "text2ecg.pt",
    "bp_translation": "bp_translation.pt",
}


def resolve_checkpoint(task, ckpt=None, revision=None, cache_dir=None):
    """Return a local filesystem path to the checkpoint for ``task``.

    * If ``ckpt`` is given it must be an existing local path (used verbatim).
    * Otherwise the canonical file (``text2ecg.pt`` / ``bp_translation.pt``) is
      downloaded from ``yang-ai-lab/SensorGen`` via ``huggingface_hub`` and its
      cached path is returned.
    """
    if task not in CHECKPOINT_FILES:
        raise ValueError(f"Unknown task '{task}'. Expected one of {list(CHECKPOINT_FILES)}.")

    if ckpt:
        if os.path.exists(ckpt):
            return ckpt
        raise FileNotFoundError(
            f"--ckpt '{ckpt}' does not exist. Omit --ckpt to download "
            f"{CHECKPOINT_FILES[task]} from Hugging Face ({HF_REPO_ID})."
        )

    filename = CHECKPOINT_FILES[task]
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "huggingface_hub is required to download checkpoints. "
            "Install it (`pip install huggingface_hub`) or pass a local --ckpt path."
        ) from exc

    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        revision=revision,
        cache_dir=cache_dir,
    )
