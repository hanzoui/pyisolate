from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from ._internal.rpc_protocol import ProxiedSingleton


class ExtensionManagerConfig(TypedDict):
    """Configuration for the :class:`ExtensionManager`.

    Controls where isolated virtual environments are created for extensions.
    """

    venv_root_path: str
    """Root directory where isolated venvs will be created (one subdir per extension)."""


class SandboxConfig(TypedDict, total=False):
    writable_paths: list[str]
    readonly_paths: list[str] | dict[str, str] # Supports src:dst mapping
    network: bool


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

    share_cuda_ipc: bool
    """If True, attempt CUDA IPC-based tensor transport (Linux only, requires ``share_torch``)."""

    sandbox: dict[str, Any]
    """Configuration for the sandbox (e.g. writable_paths, network access)."""

    env: dict[str, str]
    """Environment variable overrides for the child process."""
