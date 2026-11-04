# indic-translate-service

IndicTrans2 (`ai4bharat/indictrans2-indic-en-1B`) Indic → English translation for the Indic search
pipeline. `lighton-ocr-server` calls it once per batch of chunks of an Indic document
(`OCR_TRANSLATE_URL`), and the English goes to Vespa as `chunks_en`.

## API
`POST /translate`
```json
{"texts": ["सदर मिळकत श्री. रमेश यांच्या नावे आहे ..."], "src_lang": "mr", "fix_numbers": true}
```
→ `{"translations": ["The said property is in the name of Shri Ramesh ..."]}`, same length and order as `texts`.

- `src_lang`: `hi`, `mr`, `gu`, `bn`, `pa`; the target is always English.
- `fix_numbers` (default `true`): the number guard (`src/number_guard.py`) puts back numbers IndicTrans2 drops,
  reformats or leaves as `<ID1>` tags; Indic digits become ASCII.

`GET /health` → `{"status": "ok", "languages": [...], "model": ..., "device": ...}`

## How a chunk is translated
IndicTrans2 is trained on single sentences of at most 256 tokens, so each chunk is split into sentences
(`। ॥ . ? !`, line wraps joined, pieces capped at 60 words). All sentences of a request are sorted by
length and translated together in batches of 32 (beam 4, output ≤ 256 tokens), repetition-trimmed,
number-guarded and joined back per chunk in order. Requests are served one at a time
(IndicTransToolkit's processor isn't safe for concurrent batches).

Generation uses the KV cache. IndicTrans2's remote code predates the Transformers Cache objects and
crashed with `use_cache=True`, so the first version ran without it (every new token recomputed the
whole prefix); `translator._legacy_cache_adapter` hands the model the legacy tuple cache it expects.
`TRANSLATE_USE_CACHE=false` and `TRANSLATE_BATCH_SENTENCES` switch back / tune. `TRANSLATE_NUM_BEAMS`
(default 4) trades quality for speed: 2 is ~1.7x faster, used for bulk ingestion.

## Run
- Local / GPU host: `pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 && pip install -r requirements.txt`,
  then `make run PORT=8000`.
- Docker (model baked in, runs offline): see the header of the `Dockerfile`.
- Settings: `TRANSLATE_MODEL`, `TRANSLATE_DEVICE` (default `cuda:0` if available), `TRANSLATE_MAX_TEXTS` (64),
  `HF_HUB_OFFLINE=1` for air-gapped hosts.
- GPU memory: about 2–3 GB (fp16).
- **CPU works but is too slow for real ingestion**: measured on a 32-core host, 2 Marathi chunks
  (2,538 characters) took 468 s on CPU vs 3.4 s on GPU (≈ 140×), about 4 minutes per chunk. Generation
  runs with beam search 4 and without the KV cache (IndicTrans2's remote code breaks with the newer
  Transformers cache API), which hurts most on CPU. Use CPU only to try the API on a few chunks.

## Test
`make test` (no GPU needed; the API tests need `fastapi` and `httpx`).
