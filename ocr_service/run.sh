#!/usr/bin/env bash
# Install the OCR app when packages are missing, then start it.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"

if [[ -f "$REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO/.env"
  set +a
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

URL="http://127.0.0.1:7860/"
echo "Starting $URL"
"$ROOT/.venv/bin/uvicorn" app:app --host 0.0.0.0 --port 7860 &
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
