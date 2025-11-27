"""Utilities for sharing host path context with PyIsolate children."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

# Environment variables worth mirroring into child processes so behaviour stays
# consistent with the host interpreter.
DEFAULT_ENV_KEYS = (
    "VIRTUAL_ENV",
    "PYTHONPATH",
    "HF_HUB_DISABLE_TELEMETRY",
    "DO_NOT_TRACK",
)


def serialize_host_snapshot(
    output_path: str | os.PathLike[str] | None = None,
    extra_env_keys: Iterable[str] | None = None,
) -> dict:
    """Capture the host interpreter context for later use by child processes.

    Args:
        output_path: Optional file path to persist the snapshot as JSON.
        extra_env_keys: Additional environment variable names to capture.

    Returns:
        A dictionary with sys.path, executable, prefix, and selected env vars.
    """

    env_keys = list(DEFAULT_ENV_KEYS)
    if extra_env_keys:
        env_keys.extend(extra_env_keys)

    env = {key: os.environ[key] for key in env_keys if key in os.environ}

    snapshot = {
        "sys_path": list(sys.path),
        "sys_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "environment": env,
    }

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        snapshot["snapshot_path"] = str(path)

    return snapshot


def build_child_sys_path(
    host_paths: Sequence[str],
    extra_paths: Sequence[str],
    comfy_root: str | None = None,
) -> List[str]:
    """Construct the sys.path list a child interpreter should use.

    Host paths retain their ordering, the Comfy root is prepended when missing,
    and isolated `.venv` paths are appended while removing duplicates.
    """

    def _norm(path: str) -> str:
        return os.path.normcase(os.path.abspath(path))

    result: List[str] = []
    seen: set[str] = set()

    ordered_host = list(host_paths)
    if comfy_root:
        comfy_norm = _norm(comfy_root)
        # Remove immediate ComfyUI code subdirectories that would shadow imports
        # e.g., remove /path/to/ComfyUI/comfy, /path/to/ComfyUI/app
        # but KEEP /path/to/ComfyUI/.venv/... (site-packages needed!)
        code_subdirs = {
            os.path.join(comfy_norm, "comfy"),
            os.path.join(comfy_norm, "app"),
            os.path.join(comfy_norm, "comfy_execution"),
            # comfy_extras is a legitimate package, nodes need to import it
            os.path.join(comfy_norm, "utils"),  # Added utils to prevent shadowing
        }
        filtered_host = []
        for p in ordered_host:
            p_norm = _norm(p)
            # Skip if this is comfy_root itself (we'll add it explicitly)
            if p_norm == comfy_norm:
                continue
            # Skip if this is a known code subdirectory
            if p_norm in code_subdirs:
                continue
            filtered_host.append(p)
        ordered_host = [comfy_root] + filtered_host

    def add_path(path: str) -> None:
        if not path:
            return
        key = _norm(path)
        if key in seen:
            return
        seen.add(key)
        result.append(path)

    for path in ordered_host:
        add_path(path)

    for path in extra_paths:
        add_path(path)

    return result
