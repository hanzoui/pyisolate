"""Host-side ExtensionManager for PyIsolate.

Manages isolated virtual environments, dependency installation, and RPC lifecycle
for extensions loaded into separate processes.
"""

import logging
from typing import Generic, TypeVar, cast

from ._internal.host import Extension
from .config import ExtensionConfig, ExtensionManagerConfig
from .shared import ExtensionBase, ExtensionLocal

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=ExtensionBase)


class ExtensionManager(Generic[T]):
    """Manager for loading and supervising isolated extensions."""

    def __init__(self, extension_type: type[T], config: ExtensionManagerConfig) -> None:
        """Initialize the ExtensionManager.

        Args:
            extension_type: Base class that all managed extensions inherit from.
            config: Manager configuration (e.g., root path for virtualenvs).
        """
        self.config = config
        self.extensions: dict[str, Extension] = {}
        self.extension_type = extension_type

    def load_extension(self, config: ExtensionConfig) -> T:
        """Load an extension with the given configuration.

        Creates the venv (if isolated), installs dependencies, starts the child
        process, and returns a proxy that forwards calls to the isolated extension.
        """
        name = config["name"]
        if name in self.extensions:
            raise ValueError(f"Extension '{name}' is already loaded")

        extension = Extension(
            module_path=config["module_path"],
            extension_type=self.extension_type,
            config=config,
            venv_root_path=self.config["venv_root_path"],
        )

        self.extensions[name] = extension

        class HostExtension(ExtensionLocal):
            def __init__(self, extension_instance) -> None:
                super().__init__()
                self._extension = extension_instance
                self._proxy = None

            @property
            def proxy(self):
                if self._proxy is None:
                    if hasattr(self._extension, "ensure_process_started"):
                        self._extension.ensure_process_started()
                    self._proxy = self._extension.get_proxy()
                    self._initialize_rpc(self._extension.rpc)
                return self._proxy

            def __getattr__(self, item: str):
                if hasattr(self._extension, item):
                    return getattr(self._extension, item)
                return getattr(self.proxy, item)

        return cast(T, HostExtension(extension))
