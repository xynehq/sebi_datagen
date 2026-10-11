"""Pipeline errors.

These exceptions stop a stage instead of rewriting labels or manifests.
"""

from __future__ import annotations


class DatasetError(Exception):
    """Base error for the dataset pipeline."""


class SchemaError(DatasetError):
    """A record failed schema validation."""


class ManifestConflictError(DatasetError):
    """A manifest write would overwrite or duplicate an existing record."""


class ImmutableFieldError(DatasetError):
    """An update attempted to change a frozen provenance or label field."""


class ConfigError(DatasetError):
    """Configuration is missing, inconsistent, or unsafe."""


class GoldPolicyError(DatasetError):
    """A record or split violated the gold-label policy."""


class RankingError(DatasetError):
    """Review ranking stopped because a prediction could not be joined safely."""
