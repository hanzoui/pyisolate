from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_BLOCK_TORCH_IMPORTS = """
import builtins
_real_import = builtins.__import__

def _blocked_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("No module named 'torch'")
    return _real_import(name, *args, **kwargs)

builtins.__import__ = _blocked_import
"""


def _run_python_snippet(snippet: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(REPO_ROOT) if not existing_pythonpath else f"{REPO_ROOT}{os.pathsep}{existing_pythonpath}"
    )
    return subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )


def test_base_import_works_when_torch_is_unavailable() -> None:
    result = _run_python_snippet(
        _BLOCK_TORCH_IMPORTS
        + """
import pyisolate
print("IMPORT_OK", pyisolate.__version__)
"""
    )
    assert result.returncode == 0, result.stderr
    assert "IMPORT_OK" in result.stdout


def test_non_torch_core_api_works_when_torch_is_unavailable() -> None:
    result = _run_python_snippet(
        _BLOCK_TORCH_IMPORTS
        + """
from pyisolate import ExtensionBase, ExtensionManager, SandboxMode, singleton_scope

with singleton_scope():
    pass

manager = ExtensionManager(ExtensionBase, {"venv_root_path": "/tmp/pyisolate-venvs"})
print("CORE_OK", SandboxMode.REQUIRED.value, type(manager).__name__)
"""
    )
    assert result.returncode == 0, result.stderr
    assert "CORE_OK required ExtensionManager" in result.stdout


def test_torch_feature_raises_clear_error_when_torch_is_unavailable() -> None:
    result = _run_python_snippet(
        _BLOCK_TORCH_IMPORTS
        + """
from pyisolate._internal.tensor_serializer import register_tensor_serializer

class DummyRegistry:
    def register(self, *args, **kwargs):
        pass

try:
    register_tensor_serializer(DummyRegistry())
except RuntimeError as exc:
    print(str(exc))
    raise SystemExit(0)

raise SystemExit("Expected RuntimeError when torch is unavailable")
"""
    )
    assert result.returncode == 0, result.stderr
    assert "requires PyTorch" in result.stdout


def test_torch_feature_works_when_torch_is_available() -> None:
    pytest.importorskip("torch")
    result = _run_python_snippet(
        """
from pyisolate._internal.tensor_serializer import register_tensor_serializer

class DummyRegistry:
    def __init__(self):
        self.registered = []
    def register(self, *args, **kwargs):
        self.registered.append(args[0])

registry = DummyRegistry()
register_tensor_serializer(registry)
print("REGISTERED", len(registry.registered), "Tensor" in registry.registered)
"""
    )
    assert result.returncode == 0, result.stderr
    assert "REGISTERED" in result.stdout
