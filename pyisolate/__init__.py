"""
pyisolate - Run Python extensions in isolated virtual environments with seamless RPC.

pyisolate enables you to run Python extensions with conflicting dependencies in the
same application by automatically creating isolated virtual environments for each
extension. Extensions communicate with the host process through a transparent RPC
system, making the isolation invisible to your code.

Key Features:
    - Automatic virtual environment creation and management
    - Transparent bidirectional RPC communication
    - Zero-copy PyTorch tensor sharing (optional)
    - Shared state through ProxiedSingleton pattern
    - Support for both isolated and non-isolated extensions

Basic Usage:
    >>> import pyisolate
    >>> import asyncio
    >>> async def main():
    ...     config = pyisolate.ExtensionManagerConfig(venv_root_path="./venvs")
    ...     manager = pyisolate.ExtensionManager(pyisolate.ExtensionBase, config)
    ...     extension = await manager.load_extension(
    ...         pyisolate.ExtensionConfig(
    ...             name="my_extension",
    ...             module_path="./extensions/my_extension",
    ...             isolated=True,
    ...             dependencies=["numpy>=2.0.0"],
    ...         )
    ...     )
    ...     result = await extension.process_data([1, 2, 3])
    ...     await extension.stop()
    >>> asyncio.run(main())
"""

from ._internal.shared import ProxiedSingleton, local_execution
from .config import ExtensionConfig, ExtensionManagerConfig
from .host import ExtensionBase, ExtensionManager

__version__ = "0.0.1"

__all__ = [
    "ExtensionBase",
    "ExtensionManager",
    "ExtensionManagerConfig",
    "ExtensionConfig",
    "ProxiedSingleton",
    "local_execution",
]
