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
