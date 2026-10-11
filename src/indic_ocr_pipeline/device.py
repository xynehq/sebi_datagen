"""Pick CUDA, MPS, or CPU from the machine that is actually running."""

from __future__ import annotations


class DeviceError(RuntimeError):
    """The requested accelerator is not on this machine."""


def select_device(
    requested: str | None = "auto",
    *,
    cuda: bool | None = None,
    mps: bool | None = None,
) -> str:
    """Return ``cuda``, ``cuda:N``, ``mps``, or ``cpu``.

    ``auto`` uses CUDA when it exists, then MPS, then CPU. An explicit device
    is used only when that accelerator is present.
    """
    if cuda is None or mps is None:
        found_cuda, found_mps = _availability()
        if cuda is None:
            cuda = found_cuda
        if mps is None:
            mps = found_mps
    choice = (requested or "auto").strip().lower()
    if choice in {"", "auto"}:
        if cuda:
            return "cuda"
        if mps:
            return "mps"
        return "cpu"
    if choice == "cpu":
        return "cpu"
    if choice == "mps":
        if not mps:
            raise DeviceError("MPS was requested, but this machine does not have it.")
        return "mps"
    if choice == "cuda" or choice.startswith("cuda:"):
        if not cuda:
            raise DeviceError("CUDA was requested, but this machine does not have it.")
        return "cuda" if choice == "cuda" else choice
    raise DeviceError(f"unsupported device {requested!r}. Use auto, cuda, mps, or cpu.")


def describe_device(requested: str | None = "auto") -> str:
    """Short setup line naming the device this process will use."""
    chosen = select_device(requested)
    if not chosen.startswith("cuda"):
        if chosen == "mps":
            return "mps (Apple GPU)"
        return "cpu"
    try:
        import torch
    except ImportError:
        return chosen
    count = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0)
    return f"{chosen} ({count} CUDA device(s); first is {name})"


def _availability() -> tuple[bool, bool]:
    try:
        import torch
    except ImportError:
        return False, False
    cuda = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps = bool(mps_backend is not None and mps_backend.is_available())
    return cuda, mps
