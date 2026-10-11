"""Phase 1 command surface."""

from __future__ import annotations

from pathlib import Path

import pytest

from indic_ocr_pipeline.stages import STAGES, StageReport, main


def test_only_status_is_implemented() -> None:
    implemented = [name for name, spec in STAGES.items() if spec.implemented]
    assert implemented == [
        "ingest",
        "render",
        "detect",
        "lines",
        "ocr",
        "compare",
        "score",
        "review",
        "status",
    ]


def test_list_stages_and_unimplemented_stage() -> None:
    assert main(["--list-stages"]) == 0
    assert main(["--stage", "crops"]) == 2
    assert main(["--stage", "gold"]) == 2


def test_status_runs_against_the_repository(
    repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    del repo_root
    assert main(["--stage", "status"]) == 0
    output = capsys.readouterr().out
    assert "implemented_phase: 12" in output
    assert "human_verified and human_corrected" in output
    assert "manifest.documents:" in output
    assert "manifest.pages:" in output
    assert "manifest.regions:" in output
    assert "manifest.lines:" in output
    assert "manifest.gold: missing" in output


def test_stage_report_throughput() -> None:
    report = StageReport(
        stage="render",
        processed=4,
        completed=2,
        skipped=1,
        failed=1,
        duration_seconds=2,
    )
    assert report.throughput_per_second == 1


def test_gold_stage_is_still_unimplemented() -> None:
    assert main(["--stage", "gold"]) == 2
