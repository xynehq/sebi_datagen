"""Pipeline failures."""

from __future__ import annotations


class StageFailed(RuntimeError):
    """A critical stage stopped the pipeline."""

    def __init__(self, stage: str, message: str, module: str = "") -> None:
        self.stage = stage
        self.module = module
        super().__init__(message)
