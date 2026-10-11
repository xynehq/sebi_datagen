"""Accelerator selection follows the machine, not a hardcoded GPU."""

from __future__ import annotations

import pytest

from indic_ocr_pipeline.device import DeviceError, describe_device, select_device


def test_auto_prefers_cuda_then_mps_then_cpu() -> None:
    assert select_device("auto", cuda=True, mps=True) == "cuda"
    assert select_device("auto", cuda=False, mps=True) == "mps"
    assert select_device(None, cuda=False, mps=False) == "cpu"
    assert select_device("cuda:1", cuda=True, mps=False) == "cuda:1"


def test_explicit_device_is_rejected_when_missing() -> None:
    with pytest.raises(DeviceError, match="CUDA"):
        select_device("cuda:0", cuda=False, mps=True)
    with pytest.raises(DeviceError, match="MPS"):
        select_device("mps", cuda=True, mps=False)


def test_this_machine_resolves_to_a_real_device() -> None:
    chosen = describe_device("auto")
    assert chosen.startswith(("cuda", "mps", "cpu"))
