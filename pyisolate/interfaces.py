"""Public adapter and registry protocols for PyIsolate plugins.

These interfaces define the contract between PyIsolate core and application-
specific adapters (e.g., ComfyUI). They enable structural typing so adapters can
be implemented without inheriting from concrete base classes.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from ._internal.rpc_protocol import AsyncRPC, ProxiedSingleton


@runtime_checkable
class SerializerRegistryProtocol(Protocol):
    """Interface for dynamic type serialization registry."""

    def register(
        self,
        type_name: str,
        serializer: Callable[[Any], Any],
        deserializer: Callable[[Any], Any] | None = None,
    ) -> None:
        """Register serializer/deserializer pair for a type."""

    def get_serializer(self, type_name: str) -> Callable[[Any], Any] | None:
        """Return serializer for type if registered."""

    def get_deserializer(self, type_name: str) -> Callable[[Any], Any] | None:
        """Return deserializer for type if registered."""

    def has_handler(self, type_name: str) -> bool:
        """Return True if a serializer exists for *type_name*."""


@runtime_checkable
class IsolationAdapter(Protocol):
    """Adapter interface for application-specific isolation hooks."""

    @property
    def identifier(self) -> str:
        """Unique adapter identifier (e.g., "comfyui")."""

    def get_path_config(self, module_path: str) -> dict[str, Any] | None:
        """Compute path configuration from extension module path.

        Returns a dict with keys such as:
        - ``preferred_root``: application root directory
        - ``additional_paths``: extra sys.path entries to prepend
        """

    def setup_child_environment(self, snapshot: dict[str, Any]) -> None:
        """Configure child process environment after sys.path reconstruction."""

    def register_serializers(self, registry: SerializerRegistryProtocol) -> None:
        """Register custom type serializers for RPC transport."""

    def provide_rpc_services(self) -> list[type[ProxiedSingleton]]:
        """Return ProxiedSingleton classes to expose via RPC."""

    def handle_api_registration(self, api: ProxiedSingleton, rpc: AsyncRPC) -> None:
        """Optional post-registration hook for API-specific setup."""
