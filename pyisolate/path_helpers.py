from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

_DEFAULT_ENV_KEYS = (
    "VIRTUAL_ENV",
    "PYTHONPATH",
    "HF_HUB_DISABLE_TELEMETRY",
    "DO_NOT_TRACK",
)


def serialize_host_snapshot(
    output_path: str | os.PathLike[str] | None = None,
    extra_env_keys: Iterable[str] | None = None,
) -> dict:
    """Capture host interpreter context for child processes."""
    env_keys = list(_DEFAULT_ENV_KEYS)
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
    """Construct sys.path for child interpreter with ComfyUI root first."""

    def _norm(path: str) -> str:
        return os.path.normcase(os.path.abspath(path))

    result: List[str] = []
    seen: set[str] = set()

    ordered_host = list(host_paths)
    if comfy_root:
        comfy_norm = _norm(comfy_root)
        code_subdirs = {
            os.path.join(comfy_norm, "comfy"),
            os.path.join(comfy_norm, "app"),
            os.path.join(comfy_norm, "comfy_execution"),
            os.path.join(comfy_norm, "utils"),
        }
        filtered_host = []
        for p in ordered_host:
            p_norm = _norm(p)
            if p_norm == comfy_norm:
                continue
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
