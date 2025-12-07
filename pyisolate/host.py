import logging
from typing import Generic, TypeVar, cast

from ._internal.host import Extension
from .config import ExtensionConfig, ExtensionManagerConfig
from .shared import ExtensionBase, ExtensionLocal

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=ExtensionBase)


class ExtensionManager(Generic[T]):
    """Manages extension loading and lifecycle."""

    def __init__(self, extension_type: type[T], config: ExtensionManagerConfig) -> None:
        self.config = config
        self.extensions: dict[str, Extension] = {}
        self.extension_type = extension_type

    def load_extension(self, config: ExtensionConfig) -> T:
        """Load an extension with the given configuration."""
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
