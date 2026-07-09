"""Dataset loaders for the two released SensorGen tasks.

* :mod:`sensorgen.data.mimic_ecg`  — MIMIC-IV-ECG text-to-ECG loader.
* :mod:`sensorgen.data.vitaldb_bp` — VitalDB PPG+NIBP -> ART loader.

Both read the prebuilt native-rate HDF5 files directly; no extra data package
and no raw-data preprocessing is required at inference time.
"""
