"""Training package.

Dataset guards and checkpoint selection live here. PARSeq fine-tuning is
invoked from ``pipeline.runners`` through ``IndicPhotoOCR.recognition.finetune``.
"""

from training.dataset import assert_training_rows_safe, build_splits, load_frozen_ids
from training.selection import select_best

__all__ = [
    "assert_training_rows_safe",
    "build_splits",
    "load_frozen_ids",
    "select_best",
]
