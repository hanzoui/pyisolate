"""Dynamic serializer registry for PyIsolate plugins."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class SerializerRegistry:
    """Singleton registry for custom type serializers.

    Provides O(1) lookup for serializer/deserializer pairs registered by
    adapters. Registration occurs during bootstrap; lookups happen during
    serialization/deserialization hot paths.
    """

    _instance: SerializerRegistry | None = None

    def __init__(self) -> None:
        self._serializers: dict[str, Callable[[Any], Any]] = {}
        self._deserializers: dict[str, Callable[[Any], Any]] = {}

    @classmethod
    def get_instance(cls) -> SerializerRegistry:
        """Return the singleton instance, creating it if necessary."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(
        self,
        type_name: str,
        serializer: Callable[[Any], Any],
        deserializer: Callable[[Any], Any] | None = None,
    ) -> None:
        """Register serializer (and optional deserializer) for a type."""
        if type_name in self._serializers:
            logger.debug("Overwriting existing serializer for %s", type_name)

        self._serializers[type_name] = serializer
        if deserializer:
            self._deserializers[type_name] = deserializer
        logger.debug("Registered serializer for type: %s", type_name)

    def get_serializer(self, type_name: str) -> Callable[[Any], Any] | None:
        """Return serializer for *type_name*, or None if not registered."""
        return self._serializers.get(type_name)

    def get_deserializer(self, type_name: str) -> Callable[[Any], Any] | None:
        """Return deserializer for *type_name*, or None if not registered."""
        return self._deserializers.get(type_name)

    def has_handler(self, type_name: str) -> bool:
        """Return True if *type_name* has a registered serializer."""
        return type_name in self._serializers

    def clear(self) -> None:
        """Remove all registered handlers (useful for tests)."""
        self._serializers.clear()
        self._deserializers.clear()
