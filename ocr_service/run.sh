#!/usr/bin/env bash
# Install the OCR app when packages are missing, then start it.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"

export PATH="/opt/homebrew/bin:/usr/local/bin:${PATH}"

if [[ -f "$REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO/.env"
  set +a
fi

# Local trained LoRA (IndicPhotoOCR release copy) when the sibling repo exists.
_INDIC_ROOT="${INDICPHOTO_OCR_ROOT:-$REPO/../IndicPhotoOCR}"
_DEFAULT_MIXED_ADAPTER="${_INDIC_ROOT}/checkpoints/data/chitrapathak_marguj_mixed"
if [[ -z "${CHITRAPATHAK_MARGUJ_MIXED_ADAPTER:-}" && -d "$_DEFAULT_MIXED_ADAPTER" ]]; then
  export CHITRAPATHAK_MARGUJ_MIXED_ADAPTER="$_DEFAULT_MIXED_ADAPTER"
fi

if ! command -v tesseract >/dev/null 2>&1; then
  echo "Warning: tesseract is not on PATH. Install it (e.g. brew install tesseract tesseract-lang) for aligned Gujarati tables."
fi

env_ready() {
  "$ROOT/.venv/bin/python" - <<'PY'
import importlib.util
needed = (
    "fastapi",
    "gradio",
    "torch",
    "pypdfium2",
    "PIL",
    "yaml",
    "huggingface_hub",
    "transformers",
    "peft",
    "sentencepiece",
)
raise SystemExit(0 if all(importlib.util.find_spec(name) for name in needed) else 1)
PY
}

if [[ ! -x "$ROOT/.venv/bin/python" ]] || ! env_ready; then
  echo "Installing the OCR app."
  python3 -m venv "$ROOT/.venv"
  "$ROOT/.venv/bin/pip" install -r "$REPO/requirements.txt"
else
  echo "Environment already installed."
fi

PORT="${OCR_PORT:-7860}"
URL="http://127.0.0.1:${PORT}/"
echo "Starting $URL"
if curl -sf --max-time 2 "$URL" 2>/dev/null | grep -q "Shown even when the weights"; then
  echo "Warning: port ${PORT} is already serving an older OCR build (not this repo)."
  echo "Try: OCR_PORT=7861 ./ocr_service/run.sh"
fi
"$ROOT/.venv/bin/uvicorn" app:app --host 0.0.0.0 --port "$PORT" &
server_pid=$!
cleanup() {
  kill "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 60); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    wait "$server_pid"
    exit $?
  fi
  if curl -sf -o /dev/null "$URL"; then
    if [[ "$(uname)" == "Darwin" ]]; then
      open "$URL"
    elif command -v xdg-open >/dev/null 2>&1; then
      xdg-open "$URL" >/dev/null 2>&1 || true
    fi
    break
  fi
  sleep 0.5
done

wait "$server_pid"
