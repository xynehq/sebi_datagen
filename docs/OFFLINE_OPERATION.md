# Offline operation

After Python dependencies and local OCR weights are installed, the pipeline is intended to run with the network disabled.

## Prepare the machine

1. Install Python 3.11 or newer.
2. Install this package and PyYAML from a local wheelhouse if the machine has no network:

   ```bash
   python3 -m pip install -e ".[dev]"
   ```

   The unit tests also run without an editable install when `PYTHONPATH=src` and PyYAML and pytest are already present:

   ```bash
   PYTHONPATH=src python3 -m pytest
   ```

3. Install Tesseract and the Marathi traineddata on the machine itself.
4. Copy IndicOCR weights to a local path and set `INDICOCR_MODEL_PATH` or `configs/ocr.yaml`.
5. Copy Chitrapathak weights to a local path and set `CHITRAPATHAK_MODEL_PATH` or `configs/ocr.yaml`.
6. Set `MARATHI_OCR_DEVICE` to `cpu` or a local CUDA device such as `cuda:0`.

`.env.example` lists those variables. The program does not read the file for you and does not download weights.

## What the code refuses

- `allow_network: true` in `configs/pipeline.yaml`
- a gold status other than human verification or human correction
- treating a missing model confidence as a made-up number

No telemetry, cloud OCR, or external LLM client is part of this repository. Rendering, detection, and OCR libraries added in later phases must be imported locally and must not fetch models on first use.
