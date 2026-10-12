"""Command line for ./run_pipeline.sh."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from pipeline.fixture import prepare_fixture
from pipeline.orchestrator import execute


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="./run_pipeline.sh",
        description=(
            "Run the Indic OCR pipeline from source PDFs through frozen-benchmark "
            "evaluation. Completed stages are skipped when their inputs are unchanged."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=("dataset", "train", "eval"),
        default=None,
        help="Run one part of the pipeline. The default runs every stage.",
    )
    parser.add_argument(
        "--from",
        dest="start",
        default=None,
        help="Start at this stage and continue to the end. Example: ocr, training.",
    )
    parser.add_argument(
        "--stages",
        default=None,
        help=(
            "Comma-separated stage names to run, for example ingest,render,ocr. "
            "Export and later stages wait until review is accepted."
        ),
    )
    parser.add_argument("--list-stages", action="store_true", help="Print stage names and stop.")
    parser.add_argument(
        "--serve-review",
        action="store_true",
        help="Serve the review page so Accept, Reject, Accept all, and Reject all save.",
    )
    parser.add_argument(
        "--review-port",
        type=int,
        default=8765,
        help="Port for --serve-review. Default is 8765.",
    )
    parser.add_argument(
        "--accept-proposed",
        action="store_true",
        help="Record reports/review/decisions.json, then continue from export.",
    )
    parser.add_argument(
        "--accept-file",
        type=Path,
        default=None,
        help="Decisions JSON to record. Defaults to reports/review/decisions.json.",
    )
    parser.add_argument(
        "--reviewer",
        default="operator",
        help=(
            "Reviewer id stored with accepted lines. "
            "agent, auto, assistant, and cursor are refused."
        ),
    )
    parser.add_argument(
        "--no-continue",
        action="store_true",
        help="With --accept-proposed, record decisions and do not start the next stages.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=None,
        help="Optimizer-step budget for this training run.",
    )
    parser.add_argument("--dataset-size", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip stages already complete.")
    parser.add_argument("--check", action="store_true", help="Validate the environment and stop.")
    parser.add_argument("--dry-run", action="store_true", help="Print the stage plan and stop.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Do not contact Hugging Face or W&B.",
    )
    parser.add_argument("--language", default=None, help="Language profile, for example marathi.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Directory of source PDFs.")
    parser.add_argument("--generate", action="store_true", help="Run the dataset section only.")
    parser.add_argument(
        "--human-audit-budget",
        type=int,
        default=None,
        help="Number of lines to sample for human review.",
    )
    parser.add_argument(
        "--auto-accept-threshold",
        type=float,
        default=None,
        help="Auto-accept a class only when its precision lower bound is at least this value.",
    )
    parser.add_argument(
        "--auto-accept-min-audited",
        type=int,
        default=None,
        help="Human checks required before a category can be auto-accepted. Default is 30.",
    )
    parser.add_argument(
        "--include-auto-accepted",
        action="store_true",
        help=(
            "Add auto-accepted rows to training only. They stay gold=false and out of evaluation."
        ),
    )
    parser.add_argument(
        "--build-golden",
        action="store_true",
        help="Write data/golden from human review.",
    )
    parser.add_argument("--train", action="store_true", help="Run the training section only.")
    parser.add_argument("--eval", action="store_true", help="Run checkpoint evaluation only.")
    for index in range(1, 6):
        parser.add_argument(f"--model-{index}", default=None, help=f"Name of OCR model {index}.")
        parser.add_argument(
            f"--model-{index}-path",
            default=None,
            help=f"Local path for model {index}.",
        )
        parser.add_argument(
            f"--model-{index}-hf",
            default=None,
            help=f"Hugging Face id for model {index}, used when the local path is missing.",
        )
    parser.add_argument("--model", default=None, help="Training backend name. Default is parseq.")
    parser.add_argument(
        "--base-checkpoint",
        default=None,
        help="Local PARSeq checkpoint. Used before --base-checkpoint-hf.",
    )
    parser.add_argument(
        "--base-checkpoint-hf",
        default=None,
        help="Hugging Face id or link for the PARSeq checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="PARSeq weights. A local file if it exists, otherwise a Hugging Face link.",
    )
    parser.add_argument(
        "--indicocr",
        default=None,
        help="IndicOCR weights. A local folder if it exists, otherwise a Hugging Face link.",
    )
    parser.add_argument(
        "--chitrapathak",
        default=None,
        help="Chitrapathak weights. A local folder if it exists, otherwise a Hugging Face link.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local checkpoint or PEFT adapter. An explicit path is never replaced.",
    )
    parser.add_argument(
        "--model-hf",
        default=None,
        help="Hugging Face repo id to resolve for this run. Not downloaded during setup.",
    )
    parser.add_argument(
        "--model-url",
        default=None,
        help="Direct URL of a supported model archive or weight file.",
    )
    parser.add_argument(
        "--model-search-root",
        type=Path,
        default=None,
        help="Extra directory to search for models. The whole disk is not scanned.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List complete models in the managed directories, registry, and HF cache.",
    )
    parser.add_argument("--publish", action="store_true", help="Upload the selected checkpoint.")
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Run the isolated two-crop end-to-end fixture. No generated text.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Score a few real line crops in runs/live_smoke. "
            "Those test lines are marked human-verified only inside that folder. "
            "Trains PARSeq for one step when --base-checkpoint is set, and scores "
            "a Chitrapathak LoRA when --model-path and the official base are set."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root. Defaults to the directory that contains this package.",
    )
    args = parser.parse_args(argv)
    if args.dataset_size is not None and args.dataset_size < 1:
        print("dataset-size must be at least 1", file=sys.stderr)
        return 2
    ratios = (args.train_ratio, args.val_ratio)
    if None not in ratios and args.train_ratio + args.val_ratio >= 1:
        print("train-ratio and val-ratio must leave room for a held-out split", file=sys.stderr)
        return 2
    if args.human_audit_budget is not None and args.human_audit_budget < 1:
        print("human-audit-budget must be at least 1", file=sys.stderr)
        return 2
    if args.auto_accept_threshold is not None and not 0 <= args.auto_accept_threshold <= 1:
        print("auto-accept-threshold must be between 0 and 1", file=sys.stderr)
        return 2
    if args.auto_accept_min_audited is not None and args.auto_accept_min_audited < 1:
        print("auto-accept-min-audited must be at least 1", file=sys.stderr)
        return 2
    try:
        model_slots = _model_slots(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.publish and args.offline:
        print("--publish cannot be used with --offline", file=sys.stderr)
        return 2
    group = _stage_group(args)
    overrides: dict[str, object] = {"offline": bool(args.offline), "publish": bool(args.publish)}
    if args.dataset_size is not None:
        overrides["dataset_size"] = args.dataset_size
    if args.train_ratio is not None:
        overrides["train_ratio"] = args.train_ratio
    if args.val_ratio is not None:
        overrides["val_ratio"] = args.val_ratio
    if args.model:
        overrides["model"] = args.model
    if args.language:
        overrides["language"] = args.language
    if args.input_dir is not None:
        overrides["input_dir"] = str(args.input_dir)
    if args.human_audit_budget is not None:
        overrides["human_audit_budget"] = args.human_audit_budget
    if args.auto_accept_threshold is not None:
        overrides["auto_accept_threshold"] = args.auto_accept_threshold
    if args.auto_accept_min_audited is not None:
        overrides["auto_accept_min_audited"] = args.auto_accept_min_audited
    if args.include_auto_accepted:
        overrides["include_auto_accepted"] = True
    if args.build_golden:
        overrides["build_golden"] = True
    if model_slots:
        overrides["ocr_models"] = model_slots
    if args.base_checkpoint:
        overrides["base_checkpoint"] = args.base_checkpoint
    if args.base_checkpoint_hf:
        overrides["base_checkpoint_hf"] = args.base_checkpoint_hf
    if args.checkpoint:
        overrides["checkpoint"] = args.checkpoint
    if args.train_steps is not None:
        if args.train_steps < 1:
            print("train-steps must be at least 1", file=sys.stderr)
            return 2
        overrides["train_steps"] = args.train_steps
    if args.stages:
        try:
            from pipeline.graph import resolve_stage_name

            overrides["only_stages"] = [
                resolve_stage_name(part)
                for part in args.stages.replace(" ", ",").split(",")
                if part.strip()
            ]
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args.indicocr:
        overrides["indicocr"] = args.indicocr
    if args.chitrapathak:
        overrides["chitrapathak"] = args.chitrapathak
    repo = Path(__file__).resolve().parents[1]
    root = args.root.resolve() if args.root else repo
    if args.list_stages:
        from pipeline.graph import ORDER, STAGES

        for name in ORDER:
            print(f"{name}\t{STAGES[name].title}")
        return 0
    if args.reviewer:
        overrides["reviewer_id"] = args.reviewer
    if args.serve_review:
        from pipeline.review_pack import confirm_review

        if not confirm_review(root, port=args.review_port, reviewer_id=args.reviewer):
            print("Review was not confirmed. Training did not start.")
            return 0
        overrides["review_confirmed"] = True
        if args.start is None and args.stage is None and not args.stages:
            args.start = "export"
    if args.accept_proposed or args.accept_file is not None:
        from pipeline.accept_review import AcceptError, accept_reviews

        try:
            code = accept_reviews(
                root,
                decisions_path=args.accept_file,
                reviewer_id=args.reviewer,
            )
        except AcceptError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if code != 0:
            return code
        if args.no_continue:
            print("Decisions recorded. Continue with ./run_pipeline.sh --from export")
            return 0
        if args.start is None and args.stage is None and not args.stages:
            args.start = "export"
    if args.smoke:
        from pipeline.smoke import run_smoke

        return run_smoke(
            repo,
            adapter=args.model_path,
            official_base=args.chitrapathak,
            checkpoint=args.base_checkpoint,
            pdf=args.input_dir,
        )
    handled = _model_or_language_command(args, root)
    if handled is not None:
        return handled
    if args.fixture and (args.root is None or args.root.resolve() == repo):
        root = repo / "runs" / "fixture_workspace"
        if root.exists():
            shutil.rmtree(root)
        prepare_fixture(root)
    else:
        if args.fixture and not (root / "configs" / "pipeline.yaml").is_file():
            prepare_fixture(root)
    try:
        return execute(
            root,
            group=group,
            start=args.start,
            resume=args.resume,
            dry_run=args.dry_run,
            check_only=args.check,
            fixture=args.fixture,
            overrides=overrides,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _model_or_language_command(args: argparse.Namespace, root: Path) -> int | None:
    """Discovery, Bengali data, and PEFT adapter checks. Returns None to continue."""
    from pipeline.bengali_data import BengaliDataError, prepare
    from pipeline.discovery import (
        DiscoveryError,
        default_roots,
        discover,
        format_catalog,
        record_peft_evaluation,
        register_model,
        resolve_explicit,
        select_from_catalog,
    )
    if args.list_models:
        extra = [args.model_search_root] if args.model_search_root else None
        print(format_catalog(discover(default_roots(root, extra))))
        return 0
    if args.language == "bengali" and (args.stage == "dataset" or args.generate):
        try:
            summary = prepare(root)
        except BengaliDataError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"bengali eligible rows: {summary['eligible']}")
        return 0
    requested = _requested_model(args, root)
    if isinstance(requested, int):
        return requested
    if requested is None:
        needs_model = args.stage in {"train", "eval"} or args.eval or args.train
        if args.language not in {None, "marathi"} and needs_model and sys.stdin.isatty():
            extra = [args.model_search_root] if args.model_search_root else None
            try:
                chosen = select_from_catalog(discover(default_roots(root, extra)), interactive=True)
            except (DiscoveryError, EOFError) as exc:
                print(str(exc), file=sys.stderr)
                return 2
            requested = Path(chosen.path)
        elif args.language not in {None, "marathi"} and needs_model:
            print(
                "No model was selected. Pass --model-path or --model-hf, or run --list-models.",
                file=sys.stderr,
            )
            return 2
        else:
            return None
    try:
        model = resolve_explicit(requested)
    except DiscoveryError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    register_model(root, requested, name=model.name)
    if args.check or args.dry_run or args.stage == "dataset" or args.generate:
        return None
    if model.family == "parseq":
        return None
    if model.family == "peft" and (args.stage == "eval" or args.eval):
        language = args.language or "gujarati"
        output = root / "reports" / language / "evaluation" / "model.json"
        override = args.chitrapathak
        if override is None:
            from os import environ

            configured = environ.get("CHITRAPATHAK_MODEL_PATH", "").strip()
            override = Path(configured) if configured else None
        try:
            record_peft_evaluation(model, output, base_override=override)
        except DiscoveryError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"adapter recorded: {output}")
        print("PARSeq geometry was not applied. Metrics were not computed by this command.")
        return 0
    if model.family != "parseq" and (args.stage == "train" or args.train):
        print(
            "This model is not a PARSeq checkpoint. Chitrapathak LoRA training is not "
            "implemented in this repository. The PARSeq trainer was not called.",
            file=sys.stderr,
        )
        return 2
    if args.stage == "eval" or args.eval:
        print(
            "This model is not a PARSeq checkpoint or a resolved PEFT adapter. "
            "The PARSeq evaluator was not called.",
            file=sys.stderr,
        )
        return 2
    return None


def _requested_model(args: argparse.Namespace, root: Path) -> Path | int | None:
    from pipeline.weights import WeightError, resolve_weight

    if args.model_path is not None:
        return args.model_path.expanduser()
    remote = args.model_hf or args.model_url
    if not remote:
        return None
    if args.offline:
        print("--offline does not download models. Pass a local --model-path.", file=sys.stderr)
        return 2
    cache = root / "models" / "cache"
    expect = "dir" if args.model_hf else "file"
    try:
        resolved = resolve_weight(remote, cache, expect=expect)
    except WeightError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return Path(resolved)


def _model_slots(args: argparse.Namespace) -> list[dict[str, str | None]]:
    from pipeline.registry import parse_model_flags

    slots = []
    for index in range(1, 6):
        slots.append(
            {
                "name": getattr(args, f"model_{index}"),
                "path": getattr(args, f"model_{index}_path"),
                "hf": getattr(args, f"model_{index}_hf"),
            }
        )
    return parse_model_flags(slots)


def _stage_group(args: argparse.Namespace) -> str | None:
    selected = [args.generate or args.build_golden, args.train, args.eval]
    if sum(bool(item) for item in selected) > 1:
        return None
    if args.train:
        return "train"
    if args.eval:
        return "eval"
    if args.generate or args.build_golden:
        return "dataset"
    return args.stage


if __name__ == "__main__":
    raise SystemExit(main())
