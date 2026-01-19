from __future__ import annotations

import functools
import sys
from importlib import metadata as importlib_metadata

_CORE_TORCH_PACKAGES = frozenset({"torch", "torchvision", "torchaudio", "torchtext", "triton"})


@functools.lru_cache(maxsize=1)
def get_torch_ecosystem_packages() -> frozenset[str]:
    """Discover torch ecosystem packages present in the host environment.

    Used to skip reinstalling torch and friends when ``share_torch=True`` and the
    child inherits host site-packages.
    """
    packages: set[str] = set(_CORE_TORCH_PACKAGES)
    try:
        for dist in importlib_metadata.distributions():
            name = dist.metadata.get("Name", "").lower()
            if name.startswith(("nvidia-", "torch", "triton")):
                packages.add(name)
    except Exception:  # noqa: S110 - intentional silent fallback for metadata enumeration
        pass
    return frozenset(packages)


def probe_cuda_ipc_support() -> tuple[bool, str]:
    """Best-effort probe for CUDA IPC support on Linux.

    Returns:
        (supported, reason)
    """
    if sys.platform != "linux":
        return False, "CUDA IPC is only supported on Linux"
    try:
        import torch
    except Exception as exc:  # pragma: no cover - import guard
        return False, f"torch import failed: {exc}"

    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False"

    try:
        # Minimal handle check: event with interprocess support + tiny tensor
        torch.cuda.current_device()
        _ = torch.cuda.Event(interprocess=True)  # type: ignore[no-untyped-call]
        _ = torch.empty(1, device="cuda")
        return True, "ok"
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"CUDA IPC probe failed: {exc}"
