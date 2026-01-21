"""Adapter registry for global registration of isolation adapters."""

from __future__ import annotations

from ..interfaces import IsolationAdapter


class AdapterRegistry:
    """Singleton registry for the active isolation adapter."""

    _instance: IsolationAdapter | None = None  # noqa: UP045

    @classmethod
    def register(cls, adapter: IsolationAdapter) -> None:
        """Register adapter instance.

        Raises:
            RuntimeError: If an adapter is already registered.
        """
        if cls._instance is not None:
            # Idempotency check: if registering the exact same instance, allow it.
            if cls._instance is adapter:
                return
            raise RuntimeError(f"Adapter already registered: {cls._instance}. Call unregister() first.")
        cls._instance = adapter

    @classmethod
    def get(cls) -> IsolationAdapter | None:
        """Get registered adapter. Returns None if no adapter registered."""
        return cls._instance

    @classmethod
    def get_required(cls) -> IsolationAdapter:
        """Get adapter, raising if not registered."""
        if cls._instance is None:
            raise RuntimeError(
                "No adapter registered. Host application must call "
                "pyisolate.register_adapter() before using isolation features."
            )
        return cls._instance

    @classmethod
    def unregister(cls) -> None:
        """Clear registered adapter (for testing/cleanup)."""
        cls._instance = None
