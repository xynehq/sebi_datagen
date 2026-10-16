#!/usr/bin/env bash
# One command for the OCR dataset pipeline.
# The first run creates .venv and installs this package when the
# current Python is missing the pipeline dependencies. It does not download weights.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${ROOT}/models:${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

has_packages() {
  "$1" -c "import yaml, PIL, numpy, pypdfium2" >/dev/null 2>&1
}

if [[ -n "${PIPELINE_PYTHON:-}" ]]; then
  PY="$PIPELINE_PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
elif command -v python3.13 >/dev/null 2>&1 && has_packages "$(command -v python3.13)"; then
  PY="$(command -v python3.13)"
elif command -v python3 >/dev/null 2>&1 && has_packages "$(command -v python3)"; then
  PY="$(command -v python3)"
else
  python3 -m venv "$ROOT/.venv"
  "$ROOT/.venv/bin/python" -m pip install --upgrade pip
  "$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements.txt"
  "$ROOT/.venv/bin/python" -m pip install -e "$ROOT"
  PY="$ROOT/.venv/bin/python"
fi

exec "$PY" -m indic_ocr_pipeline "$@"
