"""Persist SensorGen inference outputs as ``.npy`` arrays plus a ``manifest.json``.

Output filenames match the arrays produced by the reference evaluation loop so
downstream tooling stays compatible:

    all_generated_samples.npy   generated signal     (N, C, T)
    all_real_samples.npy        ground-truth signal  (N, C, T)
    all_contexts.npy            dense condition      (N, C2, T)   [bp_translation]
    all_prompts.json            text prompts         list[str]    [text2ecg]
    all_nibp.npy                NIBP condition       (N, 6, 1)    [bp_translation]
    manifest.json               run metadata + array shapes
"""

import json
import os

import numpy as np


def save_results(output_dir, task, results, checkpoint=None, config=None):
    """Save arrays/prompts + a manifest under ``output_dir``. Returns the manifest."""
    os.makedirs(output_dir, exist_ok=True)
    files = {}

    def _save_npy(name, arr):
        if arr is None:
            return
        np.save(os.path.join(output_dir, name), arr)
        files[name] = list(arr.shape)

    _save_npy("all_generated_samples.npy", results.get("generated"))
    _save_npy("all_real_samples.npy", results.get("real"))
    _save_npy("all_contexts.npy", results.get("contexts"))
    _save_npy("all_nibp.npy", results.get("nibp"))

    prompts = results.get("prompts")
    if prompts:
        with open(os.path.join(output_dir, "all_prompts.json"), "w") as f:
            json.dump(prompts, f)
        files["all_prompts.json"] = len(prompts)

    generated = results.get("generated")
    manifest = {
        "task": task,
        "checkpoint": checkpoint,
        "num_samples": int(generated.shape[0]) if generated is not None else 0,
        "files": files,
        "config": config,
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest
