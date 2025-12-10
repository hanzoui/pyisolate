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
    """Dynamically discover torch ecosystem packages from the host environment.

    Queries installed packages matching ``torch*``, ``nvidia-*``, and ``triton*``
    patterns so we can safely exclude them from isolated installs when
    ``share_torch=True``.
    """
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
    """Configuration for the :class:`ExtensionManager`.

    Controls where isolated virtual environments are created for extensions.
    """

    venv_root_path: str
    """Root directory where isolated venvs will be created (one subdir per extension)."""


class ExtensionConfig(TypedDict):
    """Configuration for a single extension managed by PyIsolate."""

    name: str
    """Unique name for the extension (used for venv directory naming)."""

    module_path: str
    """Filesystem path to the extension package containing ``__init__.py``."""

    isolated: bool
    """Whether to run the extension in an isolated venv versus the host process."""

    dependencies: list[str]
    """List of pip requirement specifiers to install into the extension venv."""

    apis: list[type[ProxiedSingleton]]
    """ProxiedSingleton classes exposed to this extension for shared services."""

    share_torch: bool
    """If True, reuse host torch via torch.multiprocessing and zero-copy tensors."""
