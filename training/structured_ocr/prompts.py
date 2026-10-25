"""Inference and training prompts for structured Chitrapathak OCR."""

from __future__ import annotations

# Legacy line OCR (current sft_chitrapathak.py default)
LINE_OCR_PROMPT = (
    "Perform OCR on this image and transcribe all visible text exactly as it appears."
)

# Structured full-page target (production-aligned; train after human verification)
STRUCTURED_PAGE_PROMPT = (
    "Convert this document page to canonical markdown in correct reading order. "
    "Use # / ## / ### for headings. Use normal paragraphs for body text. "
    "Use markdown lists for bullet or numbered lists. "
    "For tables, output an HTML <table> with <thead>, <tbody>, <tr>, <th>, and <td>; "
    "put each table row's cells on one <tr> (serial, name, date, weekday together). "
    "Never list an entire column of serial numbers, then names, then dates. "
    "Transcribe in the original script and printed digits exactly. "
    "Do not transcribe decorative stamp borders or blank graphics. "
    "Stop at the end of the page."
)

SYSTEM_MESSAGE = "You are a helpful assistant."

# Reference experiment (LightOn-style render recipe — not mandatory)
LIGHTON_RENDER_DPI = 200
LIGHTON_MAX_LONGEST_EDGE = 1540
