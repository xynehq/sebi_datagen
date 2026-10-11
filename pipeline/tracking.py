"""Optional Weights & Biases logging. Missing wandb never stops a run."""

from __future__ import annotations

import os
from typing import Any


class Tracker:
    """A no-op logger unless wandb is enabled, installed, and the run is online."""

    def __init__(self, config: dict[str, Any], *, offline: bool) -> None:
        section = config.get("wandb") or {}
        self.enabled = bool(section.get("enabled")) and not offline
        self.project = str(section.get("project") or "marathi-ocr")
        self._run = None
        self.note = "disabled"
        if offline and section.get("enabled"):
            self.note = "offline; wandb not contacted"
            return
        if not self.enabled:
            return
        try:
            import wandb
        except Exception:
            self.enabled = False
            self.note = "wandb is not installed; metrics stay in the local report"
            return
        configured = section.get("mode")
        mode = "offline" if configured == "offline" else os.environ.get("WANDB_MODE", "online")
        self._run = wandb.init(project=self.project, mode=mode, config=config)
        self.note = f"wandb project {self.project} mode {mode}"

    def log(self, payload: dict[str, Any]) -> None:
        if self._run is not None:
            self._run.log(payload)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
