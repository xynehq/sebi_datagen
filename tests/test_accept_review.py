"""Review decisions become gold only after the accept command records them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pipeline.accept_review import accept_reviews
from pipeline.gold import select_gold_rows
from pipeline.orchestrator import _hold_for_review
from pipeline.review_pack import (
    ReviewSaveError,
    acceptance_text,
    save_decisions,
    write_review_pack,
)


def _consensus(tmp_path: Path) -> None:
    (tmp_path / "data" / "output").mkdir(parents=True)
    row = {
        "crop_id": "DOC_A_C1",
        "document_id": "DOC_A",
        "page_id": "DOC_A_P1",
        "line_id": "DOC_A_L1",
        "crop_path": "data/crops/a.png",
        "agreement_class": "ALL_3_AGREE",
        "predictions": {"indicocr": "अ", "chitrapathak": "अ", "tesseract": "अ"},
        "proposed_text": "अ",
    }
    (tmp_path / "data" / "output" / "consensus.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_acceptance_text_uses_the_proposed_line() -> None:
    assert acceptance_text({"proposed_text": " परीक्षा ", "predictions": {}}) == "परीक्षा"


def test_accept_records_gold_and_refuses_an_agent_reviewer(tmp_path: Path) -> None:
    _consensus(tmp_path)
    assert accept_reviews(tmp_path, reviewer_id="agent") == 2
    page = write_review_pack(tmp_path, reviewer_id="operator")
    assert page.is_file()
    stored = json.loads((tmp_path / "reports/review/decisions.json").read_text(encoding="utf-8"))
    assert stored["rows"][0]["decision"] == "none"
    assert accept_reviews(tmp_path, reviewer_id="operator") == 0
    assert not (tmp_path / "data" / "output" / "verified.jsonl").exists()
    save_decisions(
        tmp_path,
        {
            "reviewer_id": "operator",
            "rows": [{"crop_id": "DOC_A_C1", "decision": "accept", "text": "अ"}],
        },
    )
    assert accept_reviews(tmp_path, reviewer_id="operator") == 0
    verified = [
        json.loads(line)
        for line in (tmp_path / "data" / "output" / "verified.jsonl").read_text().splitlines()
    ]
    gold, rejected = select_gold_rows(verified, set())
    assert rejected == []
    assert gold[0]["gold_text"] == "अ"
    assert gold[0]["reviewer_id"] == "operator"
    names, held = _hold_for_review(tmp_path, ["preflight", "export", "train"])
    assert held is False
    assert names == ["preflight", "export", "train"]


def test_review_page_has_accept_and_reject_buttons(tmp_path: Path) -> None:
    _consensus(tmp_path)
    page = write_review_pack(tmp_path, reviewer_id="operator")
    text = page.read_text(encoding="utf-8")
    assert "Accept all" in text
    assert "Reject all" in text
    assert "Continue with dataset export, training, and evaluation?" in text
    assert 'data-action="accept"' in text
    assert 'data-action="reject"' in text


def test_save_decisions_can_reject_one_line(tmp_path: Path) -> None:
    _consensus(tmp_path)
    write_review_pack(tmp_path, reviewer_id="operator")
    save_decisions(
        tmp_path,
        {
            "reviewer_id": "operator",
            "rows": [{"crop_id": "DOC_A_C1", "decision": "reject", "text": "अ"}],
        },
    )
    payload = json.loads((tmp_path / "reports/review/decisions.json").read_text(encoding="utf-8"))
    assert payload["rows"] == [{"crop_id": "DOC_A_C1", "decision": "reject", "text": "अ"}]
    with pytest.raises(ReviewSaveError):
        save_decisions(
            tmp_path,
            {
                "reviewer_id": "operator",
                "rows": [{"crop_id": "missing", "decision": "accept", "text": "अ"}],
            },
        )


def test_review_server_saves_a_posted_decision(tmp_path: Path) -> None:
    import threading
    from urllib.request import Request, urlopen

    from pipeline.review_pack import start_review_server

    _consensus(tmp_path)
    server = start_review_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    body = json.dumps(
        {
            "reviewer_id": "operator",
            "rows": [{"crop_id": "DOC_A_C1", "decision": "reject", "text": "अ"}],
        }
    ).encode()
    request = Request(
        f"http://127.0.0.1:{port}/api/decisions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        assert response.status == 200
    saved = json.loads((tmp_path / "reports/review/decisions.json").read_text(encoding="utf-8"))
    assert saved["rows"][0]["decision"] == "reject"
    with urlopen(f"http://127.0.0.1:{port}/reports/review/index.html") as page:
        assert "Reject all" in page.read().decode("utf-8")
    server.shutdown()
    thread.join(timeout=2)


def test_continue_records_the_review_and_stops_the_server(tmp_path: Path) -> None:
    import threading
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    from pipeline.review_pack import start_review_server

    _consensus(tmp_path)
    server = start_review_server(tmp_path, port=0, reviewer_id="operator")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    pending = json.dumps(
        {
            "reviewer_id": "operator",
            "rows": [{"crop_id": "DOC_A_C1", "decision": "none", "text": "अ"}],
        }
    ).encode()
    refused = Request(
        f"http://127.0.0.1:{port}/api/continue",
        data=pending,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urlopen(refused)
    except HTTPError as exc:
        assert exc.code == 400
    else:
        raise AssertionError("unreviewed lines should not continue")
    body = json.dumps(
        {
            "reviewer_id": "operator",
            "rows": [{"crop_id": "DOC_A_C1", "decision": "accept", "text": "अ"}],
        }
    ).encode()
    request = Request(
        f"http://127.0.0.1:{port}/api/continue",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        assert response.status == 200
    thread.join(timeout=3)
    assert server.review_confirmed.is_set()
    verified = [
        json.loads(line)
        for line in (tmp_path / "data" / "output" / "verified.jsonl").read_text().splitlines()
    ]
    assert verified[0]["gold_text"] == "अ"
    server.server_close()


def test_pipeline_stops_before_export_until_review_is_accepted(tmp_path: Path) -> None:
    names, held = _hold_for_review(tmp_path, ["preflight", "calibrate", "export", "train"])
    assert held is True
    assert names == ["preflight", "calibrate"]
