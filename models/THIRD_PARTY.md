# Vendored engine

`models/IndicPhotoOCR/` is third-party recognition and detection code copied into this repository so PARSeq checkpoints can load. Its import name must stay `IndicPhotoOCR`. Checkpoint class paths use that name.

It is not a model weight directory. Weights (`*.ckpt`, `*.safetensors`, and similar) stay outside git. Licenses for bundled recognition code are under `models/IndicPhotoOCR/utils/strhub/models/`.

`models/chitrapathak/` and `models/tesseract/` are adapters written for this pipeline. They load weights from a local path or the environment.
