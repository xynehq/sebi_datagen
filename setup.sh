#!/usr/bin/env bash
# Install the environment, clear generated outputs, and run the full pipeline.
# Training starts from the published language base. A CHECKPOINT in the
# environment is ignored so a fine-tuned file is not used as the start.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

# The published language base is the training start.
unset CHECKPOINT

py_ok() {
  "$1" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
}

if [[ -n "${PIPELINE_PYTHON:-}" ]]; then
  BASE="$PIPELINE_PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  BASE="$(command -v python3)"
else
  echo "Python 3.11 or newer is required." >&2
  exit 1
fi

if ! py_ok "$BASE"; then
  echo "Python 3.11 or newer is required. Found: $("$BASE" -V 2>&1)" >&2
  exit 1
fi

env_ready() {
  "$ROOT/.venv/bin/python" - <<'PY'
import importlib.util
needed = ("yaml", "PIL", "numpy", "torch", "pypdfium2", "indic_ocr_pipeline")
raise SystemExit(0 if all(importlib.util.find_spec(name) for name in needed) else 1)
PY
}

if [[ ! -x "$ROOT/.venv/bin/python" ]] || ! env_ready; then
  if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    "$BASE" -m venv "$ROOT/.venv"
  fi
  echo "Installing the environment."
  "$ROOT/.venv/bin/python" -m pip install --upgrade pip
  "$ROOT/.venv/bin/python" -m pip install -e "$ROOT"
else
  echo "Environment already installed."
fi

"$ROOT/.venv/bin/python" - <<'PY'
import sys
print(f"python {sys.version.split()[0]}")
try:
    import torch
except Exception as exc:
    print(f"torch: not importable ({exc})")
else:
    from indic_ocr_pipeline.device import describe_device

    print(f"torch {torch.__version__}")
    print(f"device: {describe_device('auto')}")
PY

mkdir -p \
  data/pdfs \
  data/raw_ocr \
  data/calibration \
  data/golden \
  data/auto_accepted \
  data/datasets \
  manifests \
  runs \
  models/cache \
  configs/languages

missing=0
for name in INDICOCR_MODEL_PATH CHITRAPATHAK_MODEL_PATH; do
  value="${!name:-}"
  if [[ -n "$value" && ! -e "$value" ]]; then
    echo "missing ${name}: ${value}" >&2
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo "A configured local model path does not exist. Setup does not download weights." >&2
  exit 1
fi

echo "Removing generated outputs so this run starts from the PDFs."
rm -rf \
  runs \
  reports \
  lightning_logs \
  data/crops \
  data/rendered \
  data/datasets \
  data/review \
  data/calibration \
  data/golden \
  data/auto_accepted \
  data/predictions \
  data/detected \
  data/verified \
  data/gold \
  data/splits \
  data/scoring
find manifests -maxdepth 1 -type f \( -name '*.jsonl' -o -name '*.lock' -o -name '*.tmp' \) -delete

echo "Starting the pipeline from the published base model."
found_size=0
for arg in "$@"; do
  if [[ "$arg" == "--dataset-size" || "$arg" == "--datapoints" ]]; then
    found_size=1
  fi
done
if [[ "$found_size" -eq 0 ]]; then
  echo "Dataset size: every PDF."
fi
echo "The review page opens in the browser and this command waits until you confirm."
exec "$ROOT/run_pipeline.sh" --language marathi "$@"
