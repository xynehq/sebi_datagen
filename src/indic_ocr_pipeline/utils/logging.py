"""Structured logging.

Default logs stay short. ``--verbose`` enables debug records.
Counts are attached as ``key=value`` fields on the log record.
"""

from __future__ import annotations

import logging
from pathlib import Path

_EXTRA_FIELDS = (
    "processed",
    "completed",
    "skipped",
    "failed",
    "duration_seconds",
    "throughput",
)

_CONFIGURED = False


class KeyValueFormatter(logging.Formatter):
    """Format one log line with optional stage counters."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, "%Y-%m-%dT%H:%M:%S")
        message = f"{timestamp} {record.levelname} {record.name} {record.getMessage()}"
        extras = [
            f"{field}={getattr(record, field)}"
            for field in _EXTRA_FIELDS
            if hasattr(record, field)
        ]
        if extras:
            return f"{message} {' '.join(extras)}"
        return message


def configure_logging(verbose: bool = False, log_file: Path | None = None) -> None:
    """Configure the root logger once per process."""
    global _CONFIGURED
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(KeyValueFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(KeyValueFormatter())
        root.addHandler(file_handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger, configuring a default handler if needed."""
    if not _CONFIGURED and not logging.getLogger().handlers:
        configure_logging(verbose=False)
    return logging.getLogger(name)
