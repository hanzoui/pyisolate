from __future__ import annotations

import functools
import logging
from importlib import metadata as importlib_metadata
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from ._internal.shared import ProxiedSingleton

logger = logging.getLogger(__name__)

_CORE_TORCH_PACKAGES = frozenset({'torch', 'torchvision', 'torchaudio', 'torchtext', 'triton'})


@functools.lru_cache(maxsize=1)
def get_torch_ecosystem_packages() -> frozenset[str]:
    """Discover torch/nvidia/triton packages to exclude when share_torch=true."""
    packages: set[str] = set(_CORE_TORCH_PACKAGES)
    try:
        for dist in importlib_metadata.distributions():
            name = dist.metadata.get('Name', '').lower()
            if name.startswith('nvidia-') or name.startswith('torch') or name.startswith('triton'):
                packages.add(name)
    except Exception:
        pass
    return frozenset(packages)


class ExtensionManagerConfig(TypedDict):
    """Configuration for ExtensionManager."""
    venv_root_path: str


class ExtensionConfig(TypedDict):
    """Configuration for a single extension."""
    name: str
    module_path: str
    isolated: bool
    dependencies: list[str]
    apis: list[type[ProxiedSingleton]]
    share_torch: bool
