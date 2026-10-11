"""Stage registry for the dataset pipeline.

Phase 1 implements repository status only. Later phases fill in the other
stages without changing their names. OCR stages, when they exist, will write
candidates. They will not write gold labels.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from indic_ocr_pipeline import __version__
from indic_ocr_pipeline.manifests import MANIFEST_SPECS, open_manifest
from indic_ocr_pipeline.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class StageSpec:
    """One independently runnable pipeline stage."""

    name: str
    phase: int
    implemented: bool
    summary: str


@dataclass(frozen=True)
class StageReport:
    """Counters every stage prints when it finishes."""

    stage: str
    processed: int
    completed: int
    skipped: int
    failed: int
    duration_seconds: float

    @property
    def throughput_per_second(self) -> float:
        if self.duration_seconds <= 0:
            return 0.0
        return self.completed / self.duration_seconds


STAGES: dict[str, StageSpec] = {
    "ingest": StageSpec(
        "ingest",
        2,
        True,
        "Hash source PDFs and write the document manifest without modifying the files.",
    ),
    "render": StageSpec(
        "render",
        3,
        True,
        "Render PDF pages to images. The image is the visual source of truth.",
    ),
    "detect": StageSpec(
        "detect",
        4,
        True,
        "Detect text regions on rendered pages. Detection does not transcribe text.",
    ),
    "lines": StageSpec(
        "lines",
        5,
        True,
        "Detect physical text lines and write one full-line crop each. No transcription.",
    ),
    "crops": StageSpec(
        "crops",
        5,
        False,
        "Write full and partial crop images with geometry and no labels.",
    ),
    "ocr": StageSpec(
        "ocr",
        6,
        True,
        "Run local OCR backends and store candidate transcriptions. Agreement is not gold.",
    ),
    "compare": StageSpec(
        "compare",
        10,
        True,
        "Compare OCR candidates. Agreement is not ground truth.",
    ),
    "score": StageSpec(
        "score",
        11,
        True,
        "Rank candidates for review. The score is not a probability of correctness.",
    ),
    "review": StageSpec(
        "review",
        12,
        True,
        "Build the calibration batch and open human review. The reviewer is the authority.",
    ),
    "gold": StageSpec(
        "gold",
        15,
        False,
        "Copy human-verified and human-corrected rows into the gold dataset.",
    ),
    "split": StageSpec(
        "split",
        16,
        False,
        "Assign train, validation, and test by document.",
    ),
    "audit": StageSpec(
        "audit",
        17,
        False,
        "Audit leakage, labels, structure, and images.",
    ),
    "status": StageSpec(
        "status",
        1,
        True,
        "Report manifest counts and which stages are implemented.",
    ),
}


def unavailable_stage(stage: str) -> int:
    """Print why a stage cannot run yet and return exit code 2."""
    spec = STAGES[stage]
    message = (
        f"Stage {spec.name!r} is not implemented. "
        f"It is scheduled for phase {spec.phase}: {spec.summary}"
    )
    print(message, file=sys.stderr)
    logger.error(message)
    return 2


def run_status() -> int:
    """Print the phase-1 pipeline status. Missing manifests are empty, not errors."""
    from indic_ocr_pipeline.config import load_settings

    settings = load_settings()
    print(f"project: indic-ocr-pipeline {__version__}")
    implemented_phase = max(spec.phase for spec in STAGES.values() if spec.implemented)
    print(f"implemented_phase: {implemented_phase}")
    print("gold_rule: only human_verified and human_corrected samples may enter gold")
    print(f"manifests_dir: {settings.manifests_dir}")
    for name in MANIFEST_SPECS:
        manifest = open_manifest(settings.manifests_dir, name)
        if not manifest.path.exists():
            print(f"manifest.{name}: missing")
            continue
        count = len(manifest.load())
        print(f"manifest.{name}: {count}")
    print("stages:")
    for spec in STAGES.values():
        state = "ready" if spec.implemented else f"phase {spec.phase}"
        print(f"  {spec.name}: {state}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch one pipeline stage."""
    parser = argparse.ArgumentParser(
        prog="python -m indic_ocr_pipeline.stages",
        description=(
            "Run one dataset-creation stage. "
            "OCR output is a candidate. Gold text comes from human review."
        ),
    )
    parser.add_argument("--stage", choices=tuple(STAGES))
    parser.add_argument("--list-stages", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)
    if args.list_stages:
        for spec in STAGES.values():
            state = "ready" if spec.implemented else f"phase {spec.phase}"
            print(f"{spec.name}\t{state}\t{spec.summary}")
        return 0
    if args.stage is None:
        parser.error("choose --stage or --list-stages")
    if args.stage == "status":
        return run_status()
    if args.stage == "ingest":
        from indic_ocr_pipeline.pdf.ingest import main as ingest_main

        forwarded = ["--verbose"] if args.verbose else []
        return ingest_main(forwarded)
    if args.stage == "render":
        from indic_ocr_pipeline.rendering.pages import main as render_main

        forwarded = ["--verbose"] if args.verbose else []
        return render_main(forwarded)
    if args.stage == "detect":
        from indic_ocr_pipeline.detection.regions import main as detect_main

        forwarded = ["--verbose"] if args.verbose else []
        return detect_main(forwarded)
    if args.stage == "lines":
        from indic_ocr_pipeline.detection.lines import main as lines_main

        forwarded = ["--verbose"] if args.verbose else []
        return lines_main(forwarded)
    if args.stage == "ocr":
        from indic_ocr_pipeline.ocr.run import main as ocr_main

        forwarded = ["--verbose"] if args.verbose else []
        return ocr_main(forwarded)
    if args.stage in {"compare", "score"}:
        from indic_ocr_pipeline.scoring.queue import main as rank_main

        forwarded = ["--verbose"] if args.verbose else []
        return rank_main(forwarded)
    if args.stage == "review":
        from indic_ocr_pipeline.review.sample import main as sample_main

        forwarded = ["--verbose"] if args.verbose else []
        return sample_main(forwarded)
    return unavailable_stage(args.stage)


if __name__ == "__main__":
    sys.exit(main())
