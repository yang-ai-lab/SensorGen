"""SensorGen — inference code for the released SensorGen checkpoints.

This package provides code examples for running the two released SensorGen
checkpoints hosted at https://huggingface.co/yang-ai-lab/SensorGen:

* ``text2ecg``       — text-conditioned 12-lead ECG generation (MIMIC-IV-ECG).
* ``bp_translation`` — PPG + NIBP -> arterial blood-pressure reconstruction (VitalDB).

Run inference from the command line::

    python -m sensorgen.inference --task text2ecg --num-samples 64 --output-dir ./out
    python -m sensorgen.inference --task bp_translation --num-samples 64 --output-dir ./out

Checkpoints are resolved from the Hugging Face Hub repo ``yang-ai-lab/SensorGen``
(``text2ecg.pt`` / ``bp_translation.pt``) unless a local ``--ckpt`` path is given.
"""

__version__ = "0.1.0"

# The two inference tasks shipped in this release.
TASKS = ("text2ecg", "bp_translation")

__all__ = ["__version__", "TASKS"]
