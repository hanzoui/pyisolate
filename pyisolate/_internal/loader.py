"""Adapter discovery via Python entry points with fallback for source-based apps.

This module loads IsolationAdapter implementations registered under the
``pyisolate.adapters`` entry point group. For applications that run from source
(not installed as packages), it falls back to direct import.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import cast

from ..interfaces import IsolationAdapter

logger = logging.getLogger(__name__)

# Known adapters for fallback when entry points aren't available
# Format: {"name": "module.path:ClassName"}
_FALLBACK_ADAPTERS = {
    "comfyui": "comfy.isolation.adapter:ComfyUIAdapter",
}


def _try_direct_import(module_class: str) -> IsolationAdapter | None:
    """Attempt to import an adapter directly by module:class path."""
    try:
        module_path, class_name = module_class.rsplit(":", 1)
        import importlib
        module = importlib.import_module(module_path)
        adapter_cls = getattr(module, class_name)
        logger.info("📚 [PyIsolate][Loader] Direct import succeeded: %s", module_class)
        return cast(IsolationAdapter, adapter_cls())
    except Exception as exc:
        logger.debug("Direct import failed for %s: %s", module_class, exc)
        return None


def load_adapter(name: str | None = None) -> IsolationAdapter | None:
    """Load an isolation adapter by name using entry points.

    Discovery order:
    1. PYISOLATE_ADAPTER_OVERRIDE environment variable (debug override)
    2. Explicit ``name`` argument
    3. Auto-detect via entry points if exactly one adapter is installed
    4. Fallback: direct import for known adapters (source-based apps)

    Returns:
        The loaded adapter instance, or None if no adapters found.

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

    # If entry points found, use them
    if eps_list:
        if name:
            matches = [ep for ep in eps_list if ep.name == name]
            if not matches:
                available = [ep.name for ep in eps_list]
                raise ValueError(f"Adapter '{name}' not found. Available: {available}")
            if len(matches) > 1:
                raise ValueError(f"Multiple adapters registered as '{name}'")
            adapter_cls = matches[0].load()
            logger.info("📚 [PyIsolate][Loader] Loaded adapter via entry point: %s", name)
            return cast(IsolationAdapter, adapter_cls())

        if len(eps_list) == 1:
            ep = eps_list[0]
            logger.info("📚 [PyIsolate][Loader] Auto-detected adapter: %s", ep.name)
            adapter_cls = ep.load()
            return cast(IsolationAdapter, adapter_cls())

        available = [ep.name for ep in eps_list]
        raise ValueError(f"Multiple adapters found, specify one: {available}")

    # No entry points - try fallback direct import
    logger.debug("No entry points found, trying fallback imports...")

    if name:
        # Specific adapter requested
        if name in _FALLBACK_ADAPTERS:
            adapter = _try_direct_import(_FALLBACK_ADAPTERS[name])
            if adapter:
                return adapter
        raise ValueError(f"Adapter '{name}' not found via entry points or fallback.")

    # Auto-detect: try each fallback until one works
    for fallback_name, module_class in _FALLBACK_ADAPTERS.items():
        adapter = _try_direct_import(module_class)
        if adapter:
            logger.info("📚 [PyIsolate][Loader] Using fallback adapter: %s", fallback_name)
            return adapter

    logger.debug("No adapters found via entry points or fallback")
    return None
