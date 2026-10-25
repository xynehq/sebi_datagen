"""Page readings keep the markdown shape a later fine-tune will emit."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr_service"))

from reading import CHITRA_PROMPT, clean_reading, reading_markdown, reading_raw  # noqa: E402


def test_prompt_asks_for_the_stack_shape() -> None:
    assert "clean markdown" in CHITRA_PROMPT.casefold()
    assert "##" in CHITRA_PROMPT and "###" in CHITRA_PROMPT
    assert "HTML" in CHITRA_PROMPT
    assert "<table>" in CHITRA_PROMPT or "<table" in CHITRA_PROMPT
    assert "same line as the clause" in CHITRA_PROMPT
    assert "reading order" in CHITRA_PROMPT
    assert "guilloche" in CHITRA_PROMPT
    assert "Charts and graphs" in CHITRA_PROMPT


def test_structured_markdown_is_not_flattened() -> None:
    page = "\n".join(
        [
            "## दस्त गोषवारा भाग-1",
            "",
            "दस्त क्रमांक : 1940/2006",
            "",
            "| अनु क्र. | पक्षकाराचे नाव | वय |",
            "|---|---|---|",
            "| 1 | सच्चिदानंद | 42 |",
            "",
            "१३. तुम्हांस आवश्यकता भासल्यास सदरील मिळकत.",
        ]
    )
    assert reading_raw(page) == page
    assert reading_markdown(page) == page
    assert "|---|---|---|\n| 1 | सच्चिदानंद | 42 |" in reading_markdown(page)


def test_plain_lines_stay_readable() -> None:
    plain = "पहिली ओळ\nदुसरी ओळ १. पहिला मुद्दा"
    assert reading_raw(plain) == "पहिली ओळ\nदुसरी ओळ\n१. पहिला मुद्दा"
    assert reading_markdown(plain) == "पहिली ओळ\n\nदुसरी ओळ\n\n१. पहिला मुद्दा"


def test_fence_and_echoed_prompt_are_removed() -> None:
    wrapped = f"```markdown\n{CHITRA_PROMPT}\n\n## शीर्षक\n\nओळ\n```"
    assert clean_reading(wrapped, CHITRA_PROMPT) == "## शीर्षक\n\nओळ"


def test_token_and_line_loops_are_cut() -> None:
    tokens = "नाव " + " ".join(["Ri"] * 12)
    assert clean_reading(tokens).endswith("Ri Ri")
    assert clean_reading(tokens).count("Ri") == 2
    lines = "\n".join(["सही"] * 6 + ["शेवट"])
    assert clean_reading(lines) == "सही\nसही\nशेवट"


def test_block_loop_at_the_end_is_cut() -> None:
    block = "पहिली ओळ\nदुसरी ओळ"
    looped = "प्रस्तावना\n" + "\n".join([block] * 4)
    assert clean_reading(looped) == "प्रस्तावना\n" + block


def test_counting_loop_is_cut_and_real_clauses_stay() -> None:
    loop = "\n".join(f"क्रम {1000 + (10 * step)}/1978" for step in range(10))
    assert clean_reading(loop) == "क्रम 1000/1978\nक्रम 1010/1978"
    clauses = "\n".join(
        [
            "१. पहिली अट येथे लिहिली आहे",
            "२. दुसरी अट येथे लिहिली आहे",
            "३. तिसरी अट येथे लिहिली आहे",
            "४. चौथी अट येथे लिहिली आहे",
            "५. पाचवी अट येथे लिहिली आहे",
            "६. सहावी अट येथे लिहिली आहे",
            "७. सातवी अट येथे लिहिली आहे",
            "८. आठवी अट येथे लिहिली आहे",
        ]
    )
    assert clean_reading(clauses) == clauses
