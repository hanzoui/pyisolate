from __future__ import annotations

import functools
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
