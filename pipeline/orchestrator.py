"""One orchestrator for dataset creation, training, and evaluation."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from pipeline import CODE_VERSION
from pipeline.config import load_config
from pipeline.errors import StageFailed
from pipeline.fingerprints import stage_fingerprints
from pipeline.graph import DATASET_GROUP, EVAL_GROUP, ORDER, STAGES, TRAIN_GROUP, resolve_stage_name
from pipeline.preflight import checks_pass, collect_checks, format_checks
from pipeline.runinfo import GRAPH, next_run_id, publish_latest, read_latest
from pipeline.runners import Ctx, run_stage
from pipeline.state import (
    COMPLETE,
    FAILED,
    INVALIDATED,
    NOT_STARTED,
    RUNNING,
    blank_stage,
    load_state,
    now,
    save_state,
    sync_invalidation,
)
from pipeline.tracking import Tracker
from pipeline.validation import outputs_ready


def select_names(group: str | None, start: str | None) -> list[str]:
    names = list(ORDER)
    if group == "dataset":
        names = ["preflight", *DATASET_GROUP]
    elif group == "train":
        names = ["preflight", *TRAIN_GROUP]
    elif group == "eval":
        names = ["preflight", *EVAL_GROUP]
    if start:
        target = resolve_stage_name(start)
        if target not in names:
            names = [
                name
                for name in ORDER
                if name == "preflight" or ORDER.index(name) >= ORDER.index(target)
            ]
        else:
            start_index = names.index(target)
            names = [
                name
                for index, name in enumerate(names)
                if name == "preflight" or index >= start_index
            ]
    seen: list[str] = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return seen


def execute(
    root: Path,
    *,
    group: str | None = None,
    start: str | None = None,
    resume: bool = False,
    dry_run: bool = False,
    check_only: bool = False,
    fixture: bool = False,
    overrides: dict[str, Any] | None = None,
) -> int:
    del resume  # completed stages are skipped from fingerprints on every invocation
    root = root.resolve()
    config = load_config(root)
    if overrides:
        config.update(overrides)
        _apply_cli_models(config, root)
    mode = "fixture" if fixture else str((config.get("execution") or {}).get("mode", "production"))
    if check_only:
        code = _check(root, config, fixture=mode == "fixture")
        if code == 0:
            print("check: passed")
        return code
    names = select_names(group, start)
    runs_root = _output_runs(root, config)
    state_path = runs_root / "pipeline_state.json"
    state = load_state(state_path)
    latest = read_latest(runs_root)
    run_dir = latest if latest is not None and not state.get("sealed") else None
    fingerprints = stage_fingerprints(root, config, run_dir, state)
    sync_invalidation(state, fingerprints)
    fingerprints = stage_fingerprints(root, config, run_dir, state)
    plan = [
        (name, _action(name, state["stages"].get(name), fingerprints[name], root, run_dir))
        for name in names
    ]
    if dry_run:
        size = config.get("dataset_size")
        print(GRAPH)
        print()
        print(f"dataset-size: {size if size is not None else 'all crops'}")
        print("synthetic data: none")
        print("frozen-100 is scored after checkpoint selection and is not used for selection")
        for index, (name, action) in enumerate(plan, start=1):
            print(f"[{index}/{len(plan)}] {STAGES[name].title}")
            print(f"  {action}")
        return 0
    pending = [name for name, action in plan if name != "preflight" and action != "skip"]
    if not pending:
        print("Pipeline already complete. Completed stages were not rerun.")
        if latest is not None:
            print(f"latest: {latest}")
        return 0
    if run_dir is None or (state.get("sealed") and any(action == "run" for _name, action in plan)):
        run_dir = _create_run(runs_root)
        state["sealed"] = False
        fingerprints = stage_fingerprints(root, config, run_dir, state)
        sync_invalidation(state, fingerprints)
        fingerprints = stage_fingerprints(root, config, run_dir, state)
        plan = [
            (name, _action(name, state["stages"].get(name), fingerprints[name], root, run_dir))
            for name in names
        ]
    assert run_dir is not None
    log_path = run_dir / "logs" / "pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _snapshot_config(root, run_dir)
    _announce_resume(log_path, plan, state)
    tracker = Tracker(config, offline=bool(config.get("offline")))
    try:
        for index, (name, action) in enumerate(plan, start=1):
            stage = STAGES[name]
            _emit(log_path, f"[{index}/{len(plan)}] {stage.title}")
            if action == "skip":
                _emit(log_path, "  ✓ already complete")
                continue
            if name == "preflight":
                code = _check(root, config, fixture=mode == "fixture", banner=False)
                if code != 0:
                    raise StageFailed("preflight", "environment check failed", module=stage.module)
                _finish(
                    state,
                    name,
                    fingerprints[name],
                    {"messages": ["environment validated"], "row_counts": {}},
                    stage.module,
                    run_dir,
                )
                _emit(log_path, "  ✓ environment validated")
                save_state(state_path, state)
                save_state(run_dir / "state.json", state)
                continue
            if action == "adopt":
                _finish(
                    state,
                    name,
                    fingerprints[name],
                    {"messages": ["existing outputs accepted"], "row_counts": {}},
                    stage.module,
                    run_dir,
                )
                _emit(log_path, "  ✓ existing outputs accepted")
                save_state(state_path, state)
                save_state(run_dir / "state.json", state)
                continue
            record = state["stages"].setdefault(name, blank_stage(name))
            record["status"] = RUNNING
            record["started_at"] = now()
            record["module"] = stage.module
            record["errors"] = []
            save_state(state_path, state)
            result = run_stage(Ctx(root, run_dir, config, mode, name))
            fingerprints = stage_fingerprints(root, config, run_dir, state)
            _finish(state, name, fingerprints[name], result, stage.module, run_dir)
            for message in result.get("messages") or []:
                _emit(log_path, f"  ✓ {message}")
            save_state(state_path, state)
            save_state(run_dir / "state.json", state)
            tracker.log({"stage": name, "status": COMPLETE})
        if names[-1] == "report" and state["stages"].get("report", {}).get("status") == COMPLETE:
            state["sealed"] = True
            save_state(state_path, state)
            save_state(run_dir / "state.json", state)
        report = run_dir / "reports" / "final_report.html"
        if report.is_file():
            print(f"FINAL REPORT:\n{report}")
        return 0
    except StageFailed as exc:
        record = state["stages"].setdefault(exc.stage, blank_stage(exc.stage))
        record["status"] = FAILED
        record["ended_at"] = now()
        record["errors"] = [str(exc)]
        record["module"] = exc.module or STAGES[exc.stage].module
        state["sealed"] = False
        save_state(state_path, state)
        save_state(run_dir / "state.json", state)
        module = record["module"]
        recovery = STAGES[exc.stage].recovery
        text = "\n".join(
            [
                "FAILED",
                f"  stage: {exc.stage}",
                f"  module: {module}",
                f"  error: {exc}",
                f"  log: {log_path}",
                f"  recovery: {recovery}",
                "Resume with the same command. Completed stages are skipped:",
                "  ./run_pipeline.sh",
            ]
        )
        _emit(log_path, text)
        tracker.log({"stage": exc.stage, "status": FAILED})
        return 1
    finally:
        tracker.finish()


def _output_runs(root: Path, config: dict[str, Any]) -> Path:
    """Marathi keeps ``runs/``. Other languages write under their own run directory."""
    profile = config.get("language_profile") or {}
    language = str(config.get("language") or "")
    runs_dir = profile.get("runs_dir") if isinstance(profile, dict) else None
    if language and language != "marathi" and isinstance(runs_dir, str) and runs_dir:
        return root / runs_dir
    return root / "runs"


def _apply_cli_models(config: dict[str, Any], root: Path) -> None:
    training = config.setdefault("training", {})
    if config.get("base_checkpoint_hf") and not config.get("base_checkpoint"):
        training["base_checkpoint"] = str(config["base_checkpoint_hf"])
    if config.get("base_checkpoint"):
        training["base_checkpoint"] = str(config["base_checkpoint"])
    if config.get("checkpoint"):
        training["base_checkpoint"] = str(config["checkpoint"])
    models = config.setdefault("models", {})
    for name in ("indicocr", "chitrapathak"):
        if config.get(name):
            models[name] = str(config[name])
    if config.get("input_dir"):
        config.setdefault("source", {})["documents_dir"] = str(config["input_dir"])
        os.environ["OCR_INPUT_DIR"] = str(config["input_dir"])
    language = str(config.get("language") or "")
    if language:
        profile = root / "configs" / "languages" / f"{language}.yaml"
        if not profile.is_file():
            raise ValueError(f"unknown language: {language}")
        import yaml

        loaded = yaml.safe_load(profile.read_text(encoding="utf-8")) or {}
        config["language_profile"] = loaded
        if loaded.get("tesseract_lang"):
            os.environ["OCR_LANG"] = str(loaded["tesseract_lang"])


def _announce_resume(
    log_path: Path,
    plan: list[tuple[str, str]],
    state: dict[str, Any],
) -> None:
    retry = [
        name
        for name, action in plan
        if action == "run"
        and (state["stages"].get(name) or {}).get("status") in {FAILED, RUNNING}
    ]
    skipped = [name for name, action in plan if action == "skip"]
    if not retry and not skipped:
        return
    _emit(log_path, "Resuming. Completed stages are left in place.")
    if retry:
        _emit(log_path, "Retrying: " + ", ".join(retry))


def _action(
    name: str,
    record: dict[str, Any] | None,
    fingerprint: dict[str, str],
    root: Path,
    run_dir: Path | None,
) -> str:
    if name == "preflight":
        return "run"
    ready = outputs_ready(root, run_dir, name, record)
    status = None if record is None else record.get("status")
    if (
        status == COMPLETE
        and ready
        and record is not None
        and record.get("input_fingerprint") == fingerprint["input"]
        and record.get("config_hash") == fingerprint["config"]
        and record.get("output_fingerprint") == fingerprint["output"]
    ):
        return "skip"
    if status in {None, NOT_STARTED} and ready and fingerprint["output"] != "missing":
        return "adopt"
    if status == INVALIDATED:
        return "run"
    return "run"


def _finish(
    state: dict[str, Any],
    name: str,
    fingerprint: dict[str, str],
    result: dict[str, Any],
    module: str,
    run_dir: Path,
) -> None:
    record = state["stages"].setdefault(name, blank_stage(name))
    record.update(
        {
            "name": name,
            "status": COMPLETE,
            "started_at": record.get("started_at") or now(),
            "ended_at": now(),
            "input_fingerprint": fingerprint["input"],
            "output_fingerprint": fingerprint["output"],
            "config_hash": fingerprint["config"],
            "code_version": CODE_VERSION,
            "row_counts": result.get("row_counts") or {},
            "errors": [],
            "module": module,
            "artifact_root": str(run_dir),
        }
    )


def _create_run(runs_dir: Path) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / next_run_id(runs_dir)
    folders = (
        "logs",
        "config_snapshot",
        "dataset",
        "training",
        "checkpoints",
        "evaluation",
        "reports",
    )
    for relative in folders:
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    publish_latest(runs_dir, run_dir)
    return run_dir


def _snapshot_config(root: Path, run_dir: Path) -> None:
    destination = run_dir / "config_snapshot"
    destination.mkdir(parents=True, exist_ok=True)
    config_dir = root / "configs"
    if not config_dir.is_dir():
        return
    for path in config_dir.glob("*.yaml"):
        shutil.copy2(path, destination / path.name)


def _check(root: Path, config: dict[str, Any], *, fixture: bool, banner: bool = True) -> int:
    checks = collect_checks(root, config, fixture=fixture)
    if banner:
        print("[1] Preflight")
    print(format_checks(checks))
    if not checks_pass(checks):
        print("  preflight failed")
        return 1
    if banner:
        print("  ✓ environment validated")
    return 0


def _emit(log_path: Path, message: str) -> None:
    print(message, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
