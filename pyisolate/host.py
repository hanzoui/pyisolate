"""
Host implementation module for pyisolate.

This module contains the ExtensionManager class, which is the main entry point
for managing extensions across multiple virtual environments. The ExtensionManager
handles the lifecycle of extensions including creation, isolation, dependency
installation, and RPC communication setup.
"""

import logging
from typing import Generic, TypeVar, cast

from ._internal.host import Extension
from .config import ExtensionConfig, ExtensionManagerConfig
from .shared import ExtensionBase, ExtensionLocal

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=ExtensionBase)


class ExtensionManager(Generic[T]):
    """Manager for loading and managing extensions in isolated environments.

    The ExtensionManager is the primary interface for working with pyisolate.
    It handles the creation of virtual environments, installation of dependencies,
    and lifecycle management of extensions. Each extension can run in its own
    isolated environment with specific dependencies, or share the host environment.

    Type Parameters:
        T: The base type of extensions this manager will handle. Must be a subclass
            of ExtensionBase.

    Attributes:
        config: The manager configuration containing settings like venv root path.
        extensions: Dictionary mapping extension names to their Extension instances.
        extension_type: The base extension class type for all managed extensions.

    Example:
        >>> import asyncio
        >>> from pyisolate import ExtensionManager, ExtensionManagerConfig, ExtensionConfig
        >>>
        >>> async def main():
        ...     # Create manager configuration
        ...     manager_config = ExtensionManagerConfig(
        ...         venv_root_path="./my-extensions"
        ...     )
        ...
        ...     # Create manager for a specific extension type
        ...     manager = ExtensionManager(MyExtensionBase, manager_config)
        ...
        ...     # Load an extension
        ...     ext_config = ExtensionConfig(
        ...         name="processor",
        ...         module_path="./extensions/processor",
        ...         isolated=True,
        ...         dependencies=["numpy>=1.26.0"],
        ...         apis=[],
        ...         share_torch=False
        ...     )
        ...     extension = manager.load_extension(ext_config)
        ...
        ...     # Use the extension
        ...     result = await extension.process([1, 2, 3, 4, 5])
        ...     print(result)
        ...
        ...     # Clean up
        ...     await extension.stop()
        >>>
        >>> asyncio.run(main())
    """

    def __init__(self, extension_type: type[T], config: ExtensionManagerConfig) -> None:
        """Initialize the ExtensionManager.

        Args:
            extension_type: The base class that all extensions managed by this
                manager should inherit from. This is used for type checking and
                to ensure extensions have the correct interface.
            config: Configuration for the manager, including the root path for
                virtual environments.

        Raises:
            ValueError: If the venv_root_path in config is invalid or not writable.
        """
        self.config = config
        self.extensions: dict[str, Extension] = {}
        self.extension_type = extension_type

    def load_extension(self, config: ExtensionConfig) -> T:
        """Load an extension with the specified configuration.

        This method creates a new extension instance, sets up its virtual environment
        (if isolated), installs dependencies, and establishes RPC communication.
        The returned object is a proxy that forwards method calls to the extension
        running in its separate process.

        Args:
            config: Configuration for the extension, including name, module path,
                dependencies, and isolation settings.

        Returns:
            A proxy object that implements the extension interface. All async method
            calls on this object are forwarded to the actual extension via RPC.

        Raises:
            ValueError: If an extension with the same name is already loaded, or if
                the extension name or dependencies contain invalid characters.
            FileNotFoundError: If the module_path doesn't exist.
            subprocess.CalledProcessError: If dependency installation fails.
            ImportError: If the extension module cannot be imported.

        Example:
            >>> config = ExtensionConfig(
            ...     name="data_processor",
            ...     module_path="./extensions/processor",
            ...     isolated=True,
            ...     dependencies=["pandas>=2.0.0"],
            ...     apis=[DatabaseAPI],
            ...     share_torch=False
            ... )
            >>> extension = manager.load_extension(config)
            >>> # Now you can call methods on the extension
            >>> result = await extension.process_data(my_data)

        Note:
            The extension process starts immediately upon loading. To stop the
            extension and clean up resources, call the `stop()` method on the
            returned proxy object.
        """
        name = config["name"]
        if name in self.extensions:
            raise ValueError(f"Extension '{name}' is already loaded")

        logger.info(
            "📚 [PyIsolate][ExtensionManager] Loading extension name=%s module_path=%s isolated=%s share_torch=%s deps=%s",
            name,
            config.get("module_path"),
            config.get("isolated", False),
            config.get("share_torch", False),
            config.get("dependencies", []),
        )

        try:
            extension = Extension(
                module_path=config["module_path"],
                extension_type=self.extension_type,
                config=config,
                venv_root_path=self.config["venv_root_path"],
            )
        except Exception as exc:
            logger.error(
                "📚 [PyIsolate][ExtensionManager] Failed to initialize extension %s: %s",
                name,
                exc,
            )
            raise

        self.extensions[name] = extension
        proxy = extension.get_proxy()

        class HostExtension(ExtensionLocal):
            """Proxy class for the extension to provide a consistent interface.

            This internal class wraps the RPC proxy to provide the same interface
            as ExtensionBase, making remote extensions indistinguishable from
            local ones from the host's perspective.
            """

            def __init__(self, rpc, proxy, extension) -> None:
                super().__init__()
                self.proxy = proxy
                self._extension = extension

            def __getattr__(self, item: str):
                """Delegate attribute access to the extension's proxy object.

                This allows the host to call any method defined on the extension
                as if it were a local object.
                """
                return getattr(self.proxy, item)

        host_extension = HostExtension(extension.rpc, proxy, extension)
        host_extension._initialize_rpc(extension.rpc)
        logger.info(
            "📚 [PyIsolate][ExtensionManager] Extension %s ready (venv=%s)",
            name,
            extension.venv_path,
        )

        return cast(T, host_extension)

    def stop_extension(self, name: str) -> None:
        """Stop a specific extension by name.

        Args:
            name: The name of the extension to stop (as provided in ExtensionConfig).

        Raises:
            KeyError: If no extension with the given name is loaded.
        """
        if name not in self.extensions:
            raise KeyError(f"No extension named '{name}' is loaded")

        try:
            logger.info("📚 [PyIsolate][ExtensionManager] Stopping extension %s", name)
            self.extensions[name].stop()
            del self.extensions[name]
        except Exception as e:
            logger.error(f"Error stopping extension {name}: {e}")
            raise

    def stop_all_extensions(self) -> None:
        """Stop all loaded extensions and clean up resources.

        This method stops all extension processes that were loaded by this manager,
        cleaning up their virtual environments and RPC connections. It's recommended
        to call this method before shutting down the application to ensure clean
        termination of all extension processes.
        """
        for name, extension in list(self.extensions.items()):
            try:
                logger.info("📚 [PyIsolate][ExtensionManager] Stopping extension %s", name)
                extension.stop()
            except Exception as e:
                logger.error(f"Error stopping extension {name}: {e}")
        self.extensions.clear()
