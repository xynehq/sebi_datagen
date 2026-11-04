# IndicTrans2 Indic -> English translation service (GPU). The model is baked into the image so the
# container runs with no network (air-gapped): build with network, run offline.
#
#   ai4bharat/indictrans2-indic-en-1B is gated on HuggingFace: accept its terms, then build with
#   docker build --secret id=hf_token,env=HF_TOKEN -t indic-translate-service .
#   docker run --gpus all -p 8000:8000 indic-translate-service
FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --timeout 1200 --retries 10 \
    torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 1200 --retries 10 -r requirements.txt

ENV HF_HOME=/app/hf
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" python3 -c \
    "from huggingface_hub import snapshot_download; snapshot_download('ai4bharat/indictrans2-indic-en-1B')"

ENV PYTHONPATH=/app HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
COPY src/ ./

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
