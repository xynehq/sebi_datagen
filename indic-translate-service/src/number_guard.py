"""Make sure an IndicTrans2 translation keeps every number of its source sentence.

IndicTrans2 loses numbers in two ways (checked in IndicTransToolkit 1.1.1, processor.pyx):
  - IndicProcessor swaps numbers such as 11,50,000 or 5873/2019 for placeholders (<ID1>) before translating.
    The model sometimes drops the placeholder (the number vanishes) or rewrites it (`< Id1 >`, `ID1`), which
    postprocess_batch doesn't recognise, so the tag stays in the text.
  - Short numbers (e.g. "Resolution No. 14") are not protected and are sometimes just left out.

fix_numbers(source, translation) repairs one sentence:
  1. a number present with a different format (1.2 for 1.20, 1,150,000 for 11,50,000) gets the source format back;
  2. leftover placeholder tags are replaced, in order, by the source numbers still missing;
  3. any number still missing is appended at the end of the sentence in brackets: "... [11,50,000]".
Source digits in Devanagari / Gujarati / Bengali etc. are converted to ASCII 0-9 first.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

NUM = re.compile(r"\d(?:[\d,./:\-]*\d)?")
PLACEHOLDER = re.compile(r"[<\[]\s*[iI][dD]\s*\d+\s*[>\]]|\b[iI][dD]\d+\b")


def ascii_digits(s: str) -> str:
    return "".join(str(unicodedata.digit(c)) if c.isdigit() and not c.isascii() else c for c in s)


def canon(tok: str) -> str:
    """Value key: digits only, plus the decimal part without trailing zeros (1.20 == 1.2, 11,50,000 == 1,150,000)."""
    if re.fullmatch(r"\d+\.\d+", tok):
        whole, frac = tok.split(".")
        frac = frac.rstrip("0")
        return whole.lstrip("0") + ("." + frac if frac else "") or "0"
    return re.sub(r"\D", "", tok).lstrip("0") or "0"


def fix_numbers(source: str, translation: str) -> str:
    src = [t for t in NUM.findall(ascii_digits(source))]
    if not src:
        return translation
    out = translation
    have = Counter(NUM.findall(out))
    need = Counter(src)
    missing = []
    for tok, n in need.items():
        n -= min(n, have.get(tok, 0))
        if n == 0:
            continue
        # 1. same value in another format → restore the source format
        alts = [t for t in NUM.findall(out) if t != tok and canon(t) == canon(tok) and t not in need]
        for alt in alts[:n]:
            out = out.replace(alt, tok, 1)
            n -= 1
        missing += [tok] * n
    # 2. leftover placeholder tags → missing numbers in source order
    missing.sort(key=lambda t: src.index(t))
    while missing and PLACEHOLDER.search(out):
        out = PLACEHOLDER.sub(missing.pop(0), out, count=1)
    out = PLACEHOLDER.sub("", out)
    # 2b. a mangled tag after a currency sign ("₹1" from "₹<ID1>"): a 1–2 digit number the source doesn't have
    def _currency(m: re.Match) -> str:
        if missing and m.group(2) not in need:
            return m.group(1) + missing.pop(0)
        return m.group(0)
    out = re.sub(r"(₹\s?|Rs\.?\s?)(\d{1,2})\b(?![.,]\d)", _currency, out)
    # 3. anything still missing → appended to the sentence
    if missing:
        out = out.rstrip() + " [" + ", ".join(missing) + "]"
    return re.sub(r"[ ]{2,}", " ", out).strip()


if __name__ == "__main__":
    tests = [
        ("अंदाजे रु. 11,50,000 खर्चून", "at an estimated cost of Rs."),
        ("एफआयआर क्र. 0234/2022", "FIR No. < Id1 >"),
        ("एकूण क्षेत्र 1.20 हेक्टर", "Total area: 1.2 ha"),
        ("किंमत ₹48,50,000", "price of ₹1"),
        ("ठराव क्र. 14", "Resolution No."),
        ("क्षेत्र ५७.५ आर", "area 57.5 R"),
    ]
    for s, t in tests:
        print(f"{t!r:35} -> {fix_numbers(s, t)!r}")
