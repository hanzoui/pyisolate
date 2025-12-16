from __future__ import annotations

import functools
from importlib import metadata as importlib_metadata
from typing import Set

_CORE_TORCH_PACKAGES = frozenset({"torch", "torchvision", "torchaudio", "torchtext", "triton"})


@functools.lru_cache(maxsize=1)
def get_torch_ecosystem_packages() -> frozenset[str]:
    """Discover torch ecosystem packages present in the host environment.

    Used to skip reinstalling torch and friends when ``share_torch=True`` and the
    child inherits host site-packages.
    """
    packages: Set[str] = set(_CORE_TORCH_PACKAGES)
    try:
        for dist in importlib_metadata.distributions():
            name = dist.metadata.get("Name", "").lower()
            if name.startswith("nvidia-") or name.startswith("torch") or name.startswith("triton"):
                packages.add(name)
    except Exception:
        pass
    return frozenset(packages)
