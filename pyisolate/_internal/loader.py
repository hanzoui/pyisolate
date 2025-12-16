"""Adapter discovery via Python entry points.

This module loads IsolationAdapter implementations registered under the
``pyisolate.adapters`` entry point group.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import cast

from ..interfaces import IsolationAdapter

logger = logging.getLogger(__name__)


def load_adapter(name: str | None = None) -> IsolationAdapter | None:
    """Load an isolation adapter by name using entry points.

    Discovery order:
    1. PYISOLATE_ADAPTER_OVERRIDE environment variable (debug override)
    2. Explicit ``name`` argument
    3. Auto-detect if exactly one adapter is installed

    Returns:
        The loaded adapter instance, or None if no adapters installed and no name provided.

    Raises:
        ValueError: If the requested adapter is not found or discovery is ambiguous.
    """
    override = os.environ.get("PYISOLATE_ADAPTER_OVERRIDE")
    if override:
        logger.debug("Using adapter override: %s", override)
        name = override

    if sys.version_info < (3, 10):
        from importlib_metadata import entry_points
    else:
        from importlib.metadata import entry_points

    eps_obj = entry_points()
    if hasattr(eps_obj, "select"):
        eps_list = list(eps_obj.select(group="pyisolate.adapters"))
    else:
        eps_list = list(eps_obj)

    if not eps_list:
        if name:
            raise ValueError(f"Adapter '{name}' not found. No adapters installed.")
        return None

    if name:
        matches = [ep for ep in eps_list if ep.name == name]
        if not matches:
            available = [ep.name for ep in eps_list]
            raise ValueError(f"Adapter '{name}' not found. Available: {available}")
        if len(matches) > 1:
            raise ValueError(f"Multiple adapters registered as '{name}'")
        adapter_cls = matches[0].load()
        return cast(IsolationAdapter, adapter_cls())

    if len(eps_list) == 1:
        ep = eps_list[0]
        logger.info("Auto-detected adapter: %s", ep.name)
        adapter_cls = ep.load()
        return cast(IsolationAdapter, adapter_cls())

    available = [ep.name for ep in eps_list]
    raise ValueError(f"Multiple adapters found, specify one: {available}")
