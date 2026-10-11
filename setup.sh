#!/usr/bin/env bash
# Repeatable environment setup. Does not download model weights.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

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

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  "$BASE" -m venv "$ROOT/.venv"
fi
"$ROOT/.venv/bin/python" -m pip install --upgrade pip
"$ROOT/.venv/bin/python" -m pip install -e "$ROOT"

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
for name in INDICOCR_MODEL_PATH CHITRAPATHAK_MODEL_PATH CHECKPOINT; do
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

echo "setup complete"
