"""VitalDB PPG+NIBP -> ART dataset (task ``bp_translation`` / ``cross_channel``).

Reads the prebuilt native-rate HDF5 (``vitaldb_translate_nativesr.h5``) directly.
The arrays were already normalized at preprocessing time (fixed-range min-max to
``[-1, 1]`` for the waveforms; the NIBP vector is a set of pre-scaled scalars), so
no runtime normalization is applied here.

Per window the loader returns a ``(x, y, c2)`` triple mapped straight onto the
model's cross-channel interface:

    x  = art_target     (1, 1500)   generated target  (invasive ART)
    c2 = ppg_condition  (1, 1500)   dense cross-attention condition (PPG)
    y  = nibp_vector    (6, 1)      sparse AdaLN condition (NIBP summary)

HDF5 layout: ``f[split][subject][group][leaf]``; each leaf is rank-3
``(n_windows, n_ch, n_samples)`` with a float attr ``sr`` (= 50.0).
"""

import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class VitalDBBPDataset(Dataset):
    """Pre-windowed aligned-signal windows for BP reconstruction."""

    def __init__(
        self,
        h5_path: str,
        split: str = "test",
        target_sr: int = 50,
        x_signal: str = "art_target",
        c2_signal: str = "ppg_condition",
        c1_signal: str = "nibp_vector",
    ):
        super().__init__()
        self._path = h5_path
        self._target_sr = target_sr
        self._x_signal = x_signal
        self._c2_signal = c2_signal
        self._c1_signal = c1_signal
        self._h5 = None
        self._pid = None

        self._index = []
        required = [x_signal, c2_signal] + ([c1_signal] if c1_signal else [])
        with h5py.File(h5_path, "r") as f:
            if split not in f:
                raise KeyError(
                    f"Split '{split}' not in {h5_path}. Available: {list(f.keys())}"
                )
            for subj in f[split]:
                for grp in f[split][subj]:
                    sig_info = {}
                    n_examples = None
                    for sig_name in required:
                        key = f"{split}/{subj}/{grp}/{sig_name}"
                        if key not in f:
                            break
                        ds = f[key]
                        if ds.ndim == 2:
                            current, n_ch, n_samples = 1, ds.shape[0], ds.shape[1]
                            sample_idx_mode = False
                        elif ds.ndim == 3:
                            current, n_ch, n_samples = ds.shape
                            sample_idx_mode = True
                        else:
                            raise ValueError(f"Unsupported dataset rank for {key}: {ds.shape}")
                        if n_examples is None:
                            n_examples = current
                        elif current != n_examples:
                            raise ValueError(
                                f"Mismatched sample count in {grp}: {required}"
                            )
                        sig_info[sig_name] = {
                            "key": key,
                            "native_sr": float(ds.attrs["sr"]),
                            "sample_idx_mode": sample_idx_mode,
                        }
                    else:  # all required signals present -> enroll every window
                        for sample_idx in range(n_examples or 0):
                            self._index.append(
                                {"subj": subj, "grp": grp, "sample_idx": sample_idx, "sig_info": sig_info}
                            )

    def _file(self):
        # Reopen the HDF5 handle per-process so num_workers > 0 is fork-safe.
        if self._h5 is None or self._pid != os.getpid():
            self._h5 = h5py.File(self._path, "r")
            self._pid = os.getpid()
        return self._h5

    def __len__(self):
        return len(self._index)

    def _read_signal(self, entry, sig_name):
        info = entry["sig_info"][sig_name]
        ds = self._file()[info["key"]]
        raw = ds if not info["sample_idx_mode"] else ds[entry["sample_idx"]]
        return np.asarray(raw, dtype=np.float32)

    def __getitem__(self, idx):
        entry = self._index[idx]
        x_data = self._read_signal(entry, self._x_signal)                      # (1, 1500)
        c2_data = self._read_signal(entry, self._c2_signal)                    # (1, 1500)
        c1_data = self._read_signal(entry, self._c1_signal) if self._c1_signal else None  # (6, 1)
        meta = {"subject": entry["subj"], "group": entry["grp"], "window_idx": entry["sample_idx"]}
        return (
            torch.from_numpy(x_data),
            None if c1_data is None else torch.from_numpy(c1_data),
            torch.from_numpy(c2_data),
            meta,
        )


def _collate_cross_channel(batch):
    xs, c1s, c2s, metas = zip(*batch)
    return {
        "x": torch.stack(xs, dim=0),                                          # (B, 1, 1500)
        "y": None if c1s[0] is None else torch.stack(c1s, dim=0),            # (B, 6, 1)
        "c2": torch.stack(c2s, dim=0),                                        # (B, 1, 1500)
        "meta": list(metas),
    }


def build_vitaldb_bp_loader(
    h5_path: str,
    split: str = "test",
    target_sr: int = 50,
    x_signal: str = "art_target",
    c2_signal: str = "ppg_condition",
    c1_signal: str = "nibp_vector",
    batch_size: int = 32,
    num_workers: int = 4,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
) -> DataLoader:
    """Build a DataLoader yielding ``{"x", "y", "c2", "meta"}`` batches."""
    ds = VitalDBBPDataset(
        h5_path=h5_path,
        split=split,
        target_sr=target_sr,
        x_signal=x_signal,
        c2_signal=c2_signal,
        c1_signal=c1_signal,
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
        collate_fn=_collate_cross_channel,
    )
