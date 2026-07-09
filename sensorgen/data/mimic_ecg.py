"""MIMIC-IV-ECG text-to-ECG dataset (task ``text2ecg`` / ``generation``).

Reads the prebuilt native-rate HDF5 (``mimic_iv_ecg_*sr.h5``) directly and yields
``(signal[12, seg_len], text)`` pairs. The HDF5 layout (written at preprocessing
time) is::

    f[split][subject][group][leaf]   # leaf dataset shape (12, N), attrs: sr, ...
    group == f"{subject_id}_{study_id}"

Per record we (Fourier-)resample the whole waveform to ``seg_len`` samples at
``target_sr`` and per-lead min-max normalize to ``[-1, 1]`` (the fixed-range
normalization the paper found best). The text is looked up from the reports CSV
by ``{subject_id}_{study_id}``.
"""

import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.signal import resample as scipy_resample
from torch.utils.data import DataLoader, Dataset, DistributedSampler


# --------------------------------------------------------------------------- #
# Signal helpers (numerically identical to the preprocessing/eval path)
# --------------------------------------------------------------------------- #
def _ensure_finite_inplace(arr: np.ndarray) -> None:
    """Replace non-finite samples per-lead with that lead's median (else 0)."""
    if np.isfinite(arr).all():
        return
    for i in range(arr.shape[0]):
        bad = ~np.isfinite(arr[i])
        if not bad.any():
            continue
        good = ~bad
        arr[i, bad] = np.median(arr[i, good]) if good.any() else 0.0


def _minmax_normalize_2d(arr: np.ndarray) -> np.ndarray:
    """Per-lead min-max normalization to ``[-1, 1]`` (fixed-range)."""
    lead_min = arr.min(axis=1, keepdims=True)
    lead_ptp = np.ptp(arr, axis=1, keepdims=True)
    return 2.0 * ((arr - lead_min) / np.maximum(lead_ptp, 1e-8)) - 1.0


def _resample_axis1(arr: np.ndarray, n_out: int) -> np.ndarray:
    """Fourier resample along the time axis to ``n_out`` samples."""
    if arr.shape[1] == n_out:
        return arr
    return scipy_resample(arr, n_out, axis=1).astype(np.float32)


def _find_h5(h5_dir: str, name: str = "mimic_iv_ecg") -> str:
    """Locate the single ``{name}_*sr.h5`` file in ``h5_dir``."""
    cands = list(Path(h5_dir).glob(f"{name}_*sr.h5"))
    if len(cands) != 1:
        raise FileNotFoundError(
            f"Expected exactly one '{name}_*sr.h5' in {h5_dir}, found {cands}"
        )
    return str(cands[0])


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class MimicECGTextDataset(Dataset):
    """Text-conditioned 12-lead ECG windows from the MIMIC-IV-ECG HDF5."""

    def __init__(
        self,
        h5_dir: str,
        csv_path: str,
        split: str = "test",
        target_sr: int = 100,
        seg_len: int = 1000,
        text_col: str = "total_report",
        subject_col: str = "subject_id",
        study_col: str = "study_id",
        name: str = "mimic_iv_ecg",
        normalize: bool = True,
    ):
        self.path = _find_h5(h5_dir, name)
        self.target_sr = target_sr
        self.seg_len = seg_len
        self.normalize = normalize

        # Build the (h5_key, group_id, native_sr) index by scanning the tree.
        self.index = []
        with h5py.File(self.path, "r") as f:
            if split not in f:
                raise KeyError(
                    f"split '{split}' not in {self.path}; have {list(f.keys())}"
                )
            for subj in f[split]:
                for grp in f[split][subj]:
                    g = f[split][subj][grp]
                    for leaf in g:
                        if leaf.startswith("__"):        # reserved (e.g. __label__)
                            continue
                        self.index.append(
                            (f"{split}/{subj}/{grp}/{leaf}", grp, float(g[leaf].attrs["sr"]))
                        )

        # Text map: {subject_id}_{study_id} -> report string.
        df = pd.read_csv(csv_path, low_memory=False, usecols=[subject_col, study_col, text_col])
        self.text_map = {
            f"{int(r[subject_col])}_{int(r[study_col])}": (
                str(r[text_col]) if pd.notna(r[text_col]) else ""
            )
            for _, r in df.iterrows()
        }

        self._h5 = None
        self._pid = None

    def _file(self):
        # Reopen the HDF5 handle per-process so num_workers > 0 is fork-safe.
        if self._h5 is None or self._pid != os.getpid():
            self._h5 = h5py.File(self.path, "r")
            self._pid = os.getpid()
        return self._h5

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        key, group, sr = self.index[i]
        data = self._file()[key][:].astype(np.float32)        # (12, N_native)
        if self.target_sr is not None and sr != self.target_sr:
            n_out = (
                self.seg_len
                if self.seg_len is not None
                else int(data.shape[1] * self.target_sr / sr)
            )
            data = _resample_axis1(data, n_out)
        elif self.seg_len is not None and data.shape[1] != self.seg_len:
            data = data[:, : self.seg_len]
        if self.normalize:
            _ensure_finite_inplace(data)
            data = _minmax_normalize_2d(data)
        return torch.from_numpy(data), self.text_map.get(group, "")


def _collate_text(batch):
    signals, texts = zip(*batch)
    return torch.stack(signals, 0), list(texts)


def build_mimic_ecg_loader(
    h5_dir: str,
    csv_path: str,
    split: str = "test",
    batch_size: int = 32,
    num_workers: int = 4,
    target_sr: int = 100,
    seg_len: int = 1000,
    text_col: str = "total_report",
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
) -> DataLoader:
    """Build a DataLoader yielding ``(signals[B, 12, seg_len], texts[list[str]])``."""
    ds = MimicECGTextDataset(
        h5_dir=h5_dir,
        csv_path=csv_path,
        split=split,
        target_sr=target_sr,
        seg_len=seg_len,
        text_col=text_col,
    )
    sampler = (
        DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False)
        if distributed
        else None
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=_collate_text,
    )
