"""High-confidence audit sample size."""

from __future__ import annotations

from indic_ocr_pipeline.policies import high_confidence_audit_count


def test_large_pool_is_capped() -> None:
    assert high_confidence_audit_count(90_000, 0.05, 500, 1000) == 1000


def test_small_pool_is_fully_sampled() -> None:
    assert high_confidence_audit_count(10, 0.05, 500, 1000) == 10
    assert high_confidence_audit_count(0, 0.05, 500, 1000) == 0
