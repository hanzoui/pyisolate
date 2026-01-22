"""Singleton lifecycle management utilities.

This module provides context managers and utilities for managing singleton
lifecycle, particularly useful in testing scenarios where isolated singleton
scopes are needed.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any


@contextmanager
def singleton_scope() -> Generator[None, None, None]:
    """Context manager for isolated singleton scope.

    Creates an isolated scope for singletons where any singletons created
    within the scope are cleaned up on exit, and the previous singleton
    state is restored.

    This is particularly useful for:
    - Test isolation: Prevent singleton state from leaking between tests
    - Nested scopes: Allow temporary singleton overrides
    - Cleanup: Ensure singletons are properly cleaned up

    Example:
        >>> from pyisolate._internal.singleton_context import singleton_scope
        >>> from pyisolate._internal.rpc_protocol import SingletonMetaclass
        >>>
        >>> with singleton_scope():
        ...     # Any singletons created here are isolated
        ...     instance = MySingleton()
        ...     # ... use instance ...
        >>> # On exit, previous singleton state is restored

    Note:
        When using pytest-xdist (parallel tests), each worker runs in a
        separate process, so this fixture provides per-worker isolation
        automatically.
    """
    # Import here to avoid circular imports
    from .rpc_protocol import SingletonMetaclass

    # Save previous state
    previous: dict[type, Any] = SingletonMetaclass._instances.copy()
    try:
        yield
    finally:
        # Restore previous state
        SingletonMetaclass._instances.clear()
        SingletonMetaclass._instances.update(previous)
