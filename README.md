# SensorGen

Code examples for running the two released **SensorGen** checkpoints.

[![paper](https://img.shields.io/badge/paper-arXiv-b31b1b)](https://arxiv.org/abs/2607.04245)
[![Website](https://img.shields.io/badge/website-SensorGen-blue)](https://yang-ai-lab.github.io/sensor-gen)
[![GitHub](https://img.shields.io/badge/code-GitHub-181717?logo=github)](https://github.com/yang-ai-lab/SensorGen)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-SensorGen-FFD21E)](https://huggingface.co/yang-ai-lab/SensorGen)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)


## 🔥 News

- [2026-07-05] Paper is now available: **[Signal or Noise? Understanding Generative Models for Real-World Sensor Time Series](https://arxiv.org/abs/2607.04245)**.

- [2026-07-03] Initial open release of the two **SensorGen** checkpoints: **Text-to-ECG** generation (MIMIC-IV ECG) and **Invasive BP Reconstruction** (VitalDB).


## 📖 Introduction

This repository provides code examples for running the two released SensorGen
checkpoints hosted at https://huggingface.co/yang-ai-lab/SensorGen.

The released checkpoints support:

1. **Text-to-ECG generation** on MIMIC-IV ECG.
2. **Invasive blood pressure reconstruction** on VitalDB.


## 📦 Released Checkpoints

| Checkpoint | Task | Dataset | Input | Output |
| ---------- | ---- | ------- | ----- | ------ |
| text2ecg.pt | Text-to-ECG | MIMIC-IV ECG | Free-text ECG report | 12-lead ECG |
| bp_translation.pt | BP reconstruction | VitalDB | PPG waveform + NIBP summary | arterial blood pressure waveform |

Both checkpoints are hosted on the Hugging Face Hub repo
[`yang-ai-lab/SensorGen`](https://huggingface.co/yang-ai-lab/SensorGen) and are
downloaded automatically on first use. To fetch a specific file:

```python
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download(
    repo_id="yang-ai-lab/SensorGen",
    filename="text2ecg.pt",
)

ckpt_path = hf_hub_download(
    repo_id="yang-ai-lab/SensorGen",
    filename="bp_translation.pt",
)
```


## 📖 Table of Contents

1. [Installation](#-installation)
2. [Quick Start](#-quick-start)
3. [Usage](#-usage)
4. [Datasets](#-datasets)
5. [Data Paths & HDF5 Layout](#-data-paths--hdf5-layout)
6. [Project Structure](#-project-structure)
7. [Citation](#-citation)
8. [License](#-license)


## 💿 Installation

```bash
# Clone this repository and enter the project directory.
conda create -n sensorgen python=3.12 -y
conda activate sensorgen
pip install torch torchvision torchaudio    # match your CUDA build (Hopper / GH200, ...)
pip install -r requirements.txt
```

### Dependencies

- Python >= 3.10
- PyTorch >= 2.4 (Hopper / GH200 builds recommended)
- `huggingface_hub` (checkpoint download), `h5py`, `scipy`, `pandas`, `PyYAML`, `timm`, `torchdiffeq`
- Diffusers, Transformers, SafeTensors (for the text encoder used by the Text-to-ECG task)


## 🚀 Quick Start

The quickest way to see both released checkpoints in action is the demo notebook:

**[`examples/SensorGen_inference_demo.ipynb`](examples/SensorGen_inference_demo.ipynb)**

To reproduce with your own prepared data, set `RUN_MODEL=True` (or use the CLI below).

To run inference from the command line on the released Text-to-ECG checkpoint
(single GPU or CPU):

```bash
python -m sensorgen.inference \
  --task text2ecg \
  --checkpoint ./ckpts/text2ecg.pt \
  --config configs/text2ecg.yaml \
  --output_dir outputs/text2ecg
```

This loads the checkpoint weights, samples on the MIMIC-IV test-split reports,
and writes the generated 12-lead ECG signals to `outputs/text2ecg/`. If
`--checkpoint` is omitted, `text2ecg.pt` is downloaded from
Hugging Face automatically. Point the dataset paths in `configs/text2ecg.yaml`
(`h5_dir`, `csv_path`) at your local MIMIC-IV ECG copy first — see
[Data Paths & HDF5 Layout](#-data-paths--hdf5-layout).


## 👩‍💻 Usage

Run inference for either released task; generated samples are written to
`--output_dir`.

```bash
# Text-to-ECG
python -m sensorgen.inference \
  --task text2ecg \
  --checkpoint ./ckpts/text2ecg.pt \
  --config configs/text2ecg.yaml \
  --output_dir outputs/text2ecg

# Invasive BP reconstruction
python -m sensorgen.inference \
  --task bp_translation \
  --checkpoint ./ckpts/bp_translation.pt \
  --config configs/bp_translation.yaml \
  --output_dir outputs/bp_translation
```

Common flags: `--num-samples`, `--batch-size`, `--num-workers`, `--split`,
`--device`, `--seed`, and `--cfg-scale` (classifier-free guidance; `> 1.0`
enables CFG for Text-to-ECG). Run `python -m sensorgen.inference --help` for the
full list.


## 📊 Datasets

Neither MIMIC-IV ECG nor VitalDB are redistributed in this repository.
Obtain credentialed access from the original sources and preprocess into the
HDF5 layout below before running inference.

| Dataset                   | Source                                                                    | Used for                   |
| ------------------------- | ------------------------------------------------------------------------- | -------------------------- |
| MIMIC-IV ECG              | [PhysioNet — MIMIC-IV ECG](https://physionet.org/content/mimic-iv-ecg/)   | Text-to-ECG                |
| VitalDB (translate split) | [VitalDB](https://vitaldb.net)                                            | Invasive BP Reconstruction |


## 🔧 Data Paths & HDF5 Layout

The two tasks read their inputs directly from the `h5_dir` / `h5_path` /
`csv_path` entries in the YAML configs under `configs/`. Point those at your
local copies before sampling:

```
your_path_to_universal_signal_dataset/
├── mimic_iv/processed/mimic_iv_ecg_nativesr.h5
│   layout: <split>/<subject_id>/<subject_id>_<study_id>/<leaf>
│           leaf dataset shape (12, N), attrs: sr
└── vitaldb_translate/vitaldb_translate_nativesr.h5
    layout: <split>/<case_id>/<group>/{art_target, ppg_condition, nibp_vector}
            art_target / ppg_condition (n_windows, 1, 1500); nibp_vector (n_windows, 6, 1); attrs: sr

your_path_to_mimic_iv_ecg/
└── preprocessed_reports.csv
    columns: subject_id, study_id, total_report
```

Edit the relevant keys in each config:

- `configs/text2ecg.yaml` — `h5_dir`, `csv_path`
- `configs/bp_translation.yaml` — `h5_path`


## 📁 Project Structure

```
SensorGen/
├── README.md
├── sensorgen/                       # Inference package (public API)
│   ├── inference.py                 # CLI entry: python -m sensorgen.inference
│   ├── checkpoint.py                # Hugging Face checkpoint resolver (yang-ai-lab/SensorGen)
│   ├── output.py                    # .npy / manifest writer
│   └── data/
│       ├── mimic_ecg.py             # MIMIC-IV ECG text-to-ECG loader
│       └── vitaldb_bp.py            # VitalDB PPG+NIBP -> ART loader
├── configs/
│   ├── text2ecg.yaml
│   └── bp_translation.yaml
├── Model/                           # model implementation
├── examples/
│   └── SensorGen_inference_demo.ipynb   # executed demo
├── requirements.txt
└── LICENSE
```

Checkpoints are hosted externally on Hugging Face — see [Released Checkpoints](#-released-checkpoints).


## 📝 Citation

If you use this code or any of the released checkpoints, please cite
the SensorGen paper (preprint forthcoming):

```bibtex
@article{shuai2026sensorgen,
  title={Signal or Noise? Understanding Generative Models for Real-World
         Sensor Time Series},
  author={Shuai, Zitao and Xu, Zongzhe and Wu, Yuntian and Li, Sirui and
          Li, Tianhong and Yang, Yuzhe},
  journal={arXiv preprint},
  year={2026}
}
```


## 📜 License

This release is distributed under the MIT License (see the top-level `LICENSE`).
