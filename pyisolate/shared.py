"""Public host/extension shared interfaces for PyIsolate."""

from types import ModuleType
from typing import TypeVar, final

from ._internal.shared import AsyncRPC, ProxiedSingleton

proxied_type = TypeVar("proxied_type", bound=object)


class ExtensionLocal:
    """Base class for code that runs inside the extension process."""

    async def before_module_loaded(self) -> None:
        """Hook called before the extension module is imported."""

    async def on_module_loaded(self, module: ModuleType) -> None:
        """Hook called after the extension module is successfully loaded."""

    @final
    def _initialize_rpc(self, rpc: AsyncRPC) -> None:
        """Initialize RPC communication (called internally by the framework)."""
        self._rpc = rpc

    @final
    def register_callee(self, object_instance: object, object_id: str) -> None:
        """Expose an object for remote calls from the host process."""
        self._rpc.register_callee(object_instance, object_id)

    @final
    def create_caller(self, object_type: type[proxied_type], object_id: str) -> proxied_type:
        """Create a proxy for calling methods on a remote object."""
        return self._rpc.create_caller(object_type, object_id)

    @final
    def use_remote(self, proxied_singleton: type[ProxiedSingleton]) -> None:
        """Configure a ProxiedSingleton class to resolve to remote instances."""
        proxied_singleton.use_remote(self._rpc)


class ExtensionBase(ExtensionLocal):
    """Base class for all PyIsolate extensions, providing lifecycle hooks and RPC wiring."""

    def __init__(self) -> None:
        super().__init__()

    async def stop(self) -> None:
        """Stop the extension and clean up resources."""
        await self._rpc.stop()
