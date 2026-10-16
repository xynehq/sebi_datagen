"""Orchestrator checks, dry-run, and a two-crop fixture."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from pipeline.dataset_flow import build_consensus, export_datasets
from pipeline.gold import ensure_invalidated_archive


def test_check_and_dry_run(repo_root: Path) -> None:
    check = subprocess.run(
        ["./run_pipeline.sh", "--check"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stdout + check.stderr
    assert "check: passed" in check.stdout

    dry = subprocess.run(
        ["./run_pipeline.sh", "--dry-run", "--dataset-size", "10000"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert dry.returncode == 0, dry.stdout + dry.stderr
    assert "dataset-size: 10000" in dry.stdout
    assert "synthetic data: none" in dry.stdout
    assert "frozen-100" in dry.stdout


def test_fixture_counts_and_invalidated_review(tmp_path: Path) -> None:
    manifests = tmp_path / "data" / "output"
    manifests.mkdir(parents=True)
    crops = [
        _crop("D1", 1),
        _crop("D1", 2),
    ]
    predictions = []
    for crop in crops:
        for backend in ("indicocr", "chitrapathak", "tesseract"):
            predictions.append(
                {
                    "prediction_id": f"{crop['crop_id']}_{backend}",
                    "crop_id": crop["crop_id"],
                    "backend": backend,
                    "text": "महाराष्ट्र",
                    "status": "complete",
                }
            )
    _write(manifests / "crops.jsonl", crops)
    _write(manifests / "predictions.jsonl", predictions)
    rows = build_consensus(tmp_path)
    assert len(rows) == 2
    assert all(row["gold"] is False for row in rows)
    assert all(row["status"] == "AUTO_3_AGREE" for row in rows)

    verified = [
        {
            "crop_id": "D1_P0001_L0001_C_full_01",
            "verified_id": "old",
            "document_id": "D1",
            "page_id": "D1_P0001",
            "parent_line_id": "D1_P0001_L0001",
            "reviewer_id": "local_reviewer",
            "verification_status": "human_verified",
            "review_decision": "accept_candidate",
            "final_text": "महाराष्ट्र",
            "verification_timestamp": "2026-10-07T00:00:00Z",
        }
    ]
    _write(manifests / "verified.jsonl", verified)
    invalidated = ensure_invalidated_archive(tmp_path)
    assert "D1_P0001_L0001_C_full_01" in invalidated
    summary = export_datasets(tmp_path, {"paths": {}, "seed": 1})
    assert summary["gold"] == 0
    assert summary["invalidated"] == 1
    train = (tmp_path / "data" / "datasets" / "train.jsonl").read_text(encoding="utf-8")
    assert train == ""


def _crop(document: str, index: int) -> dict[str, object]:
    crop_id = f"{document}_P0001_L{index:04d}_C_full_01"
    return {
        "crop_id": crop_id,
        "document_id": document,
        "page_id": f"{document}_P0001",
        "parent_line_id": f"{document}_P0001_L{index:04d}",
        "image_path": f"data/crops/{crop_id}.png",
        "status": "complete",
    }


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_publish_refused_offline() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src"), str(root / "models"), str(root), env.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(
        [sys.executable, "-m", "pipeline", "--publish", "--offline", "--dry-run"],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "offline" in completed.stderr
