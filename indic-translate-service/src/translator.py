"""IndicTrans2 Indic → English, chunk in, chunk out.

Every chunk of a request is split into sentences (the model is trained on single sentences). All
the request's sentences are then sorted by length and translated together in batches with beam
search (fuller batches, less padding), repetition-trimmed, number-guarded, and joined back per
chunk in their original order, so the English stays aligned with the original chunks.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from number_guard import fix_numbers
from text import split_sentences, trim_repeats

# lighton-ocr-server sends these short codes (translate_client.SERVICE_LANGUAGE).
FLORES_CODE = {
    "hi": "hin_Deva",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "bn": "ben_Beng",
    "pa": "pan_Guru",
}
SENTENCES_PER_BATCH = int(os.environ.get("TRANSLATE_BATCH_SENTENCES", "32"))
MAX_OUTPUT_TOKENS = 256
# Beam width: 4 is the IndicTrans2 default; 2 is ~1.7x faster for a small quality cost (bulk ingestion).
NUM_BEAMS = max(1, int(os.environ.get("TRANSLATE_NUM_BEAMS", "4")))
# The KV cache makes generation several times faster; see _legacy_cache_adapter.
USE_CACHE = os.environ.get("TRANSLATE_USE_CACHE", "true").lower() not in ("0", "false", "no")


def group_sentences(texts: list[str]) -> tuple[list[str], list[int]]:
    """All sentences of all texts, and for each sentence the index of the text it came from."""
    sentences: list[str] = []
    owners: list[int] = []
    for i, text in enumerate(texts):
        for sentence in split_sentences(text):
            sentences.append(sentence)
            owners.append(i)
    return sentences, owners


def length_batches(sentences: list[str], size: int) -> list[list[int]]:
    """Sentence indices in batches of similar length (less padding per batch)."""
    order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
    return [order[i : i + size] for i in range(0, len(order), size)]


def join_by_owner(translations: list[str], owners: list[int], n_texts: int) -> list[str]:
    parts: list[list[str]] = [[] for _ in range(n_texts)]
    for owner, text in zip(owners, translations):
        parts[owner].append(text)
    return [" ".join(p) for p in parts]


def _legacy_cache_adapter(model: Any) -> None:
    """Let IndicTrans2's remote code use the KV cache with Transformers ≥ 4.5x.

    Its decoder expects the old tuple cache (`past_key_values[0][0].shape`), but `generate` now
    passes a Cache object (empty on the first step), which crashed it, so the service used to run
    with use_cache=False: every new token recomputed the whole prefix. Here each step's cache is
    handed to the model as the legacy tuple (None on the first step); `generate` keeps managing its
    own Cache object, including beam reordering.
    """
    from transformers.cache_utils import Cache

    original = model.prepare_inputs_for_generation

    def prepare(*args: Any, **kwargs: Any) -> dict[str, Any]:
        inputs = original(*args, **kwargs)
        past = inputs.get("past_key_values")
        if isinstance(past, Cache):
            inputs["past_key_values"] = None if past.get_seq_length() == 0 else past.to_legacy_cache()
        return inputs

    model.prepare_inputs_for_generation = prepare


class Translator:
    def __init__(self, model_name: str, device: str) -> None:
        import torch
        from IndicTransToolkit.processor import IndicProcessor
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.model_name = model_name
        self.device = device
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=dtype)
            .to(device)
            .eval()
        )
        if USE_CACHE:
            _legacy_cache_adapter(self.model)
        self.processor = IndicProcessor(inference=True)
        # One request at a time: IndicProcessor keeps a batch's placeholder maps in one shared
        # queue, so concurrent requests would mix up (or wait forever for) each other's maps.
        self._lock = threading.Lock()

    def translate(self, texts: list[str], src_lang: str, *, fix_numbers_on: bool = True) -> list[str]:
        src = FLORES_CODE[src_lang]
        sentences, owners = group_sentences(texts)
        if not sentences:
            return [""] * len(texts)
        english: list[str] = [""] * len(sentences)
        with self._lock:
            for idx in length_batches(sentences, SENTENCES_PER_BATCH):
                batch = [sentences[i] for i in idx]
                for i, out in zip(idx, self._translate_batch(batch, src, fix_numbers_on=fix_numbers_on)):
                    english[i] = out
        return join_by_owner(english, owners, len(texts))

    def _translate_batch(self, sentences: list[str], src: str, *, fix_numbers_on: bool) -> list[str]:
        prepared = self.processor.preprocess_batch(sentences, src_lang=src, tgt_lang="eng_Latn")
        enc = self.tokenizer(
            prepared,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)
        max_len = min(MAX_OUTPUT_TOKENS, int(enc["input_ids"].shape[1] * 1.6) + 16)
        with self._torch.inference_mode():
            generated = self.model.generate(
                **enc,
                use_cache=USE_CACHE,
                num_beams=NUM_BEAMS,
                max_length=max_len,
                no_repeat_ngram_size=4,
                num_return_sequences=1,
            )
        decoded = self.tokenizer.batch_decode(
            generated.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        out = self.processor.postprocess_batch(decoded, lang="eng_Latn")
        # number guard: put back numbers IndicTrans2 dropped or left as <ID1> tags
        return [
            fix_numbers(source, trim_repeats(target)) if fix_numbers_on else trim_repeats(target)
            for source, target in zip(sentences, out)
        ]

    def health(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "model": self.model_name,
            "device": self.device,
            "use_cache": USE_CACHE,
            "batch_sentences": SENTENCES_PER_BATCH,
            "num_beams": NUM_BEAMS,
        }
        if self.device.startswith("cuda"):
            info["gpu_gb"] = round(self._torch.cuda.memory_allocated() / 1e9, 2)
        return info
