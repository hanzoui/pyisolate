import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pyisolate._internal import host
from pyisolate._internal.host import (
    _DeduplicationFilter,
    _probe_cuda_ipc_support,
    environment,
    normalize_extension_name,
    validate_dependency,
    validate_path_within_root,
    Extension,
)
from pyisolate.config import ExtensionConfig


def make_extension(tmp_path, share_torch: bool = True) -> Extension:
    config: ExtensionConfig = {
        "name": "demo",
        "dependencies": [],
        "share_torch": share_torch,
        "share_cuda_ipc": False,
        "apis": [],
    }
    return Extension(module_path="/tmp/mod.py", extension_type=SimpleNamespace, config=config, venv_root_path=str(tmp_path))


def test_normalize_extension_name_valid():
    assert normalize_extension_name("abc") == "abc"
    assert normalize_extension_name("../weird") == "weird"
    assert normalize_extension_name("a b!c") == "a_b_c"


def test_normalize_extension_name_empty_raises():
    with pytest.raises(ValueError):
        normalize_extension_name("")


@pytest.mark.parametrize(
    "dep",
    ["-e", "-e .", "torch==2.2.0", "numpy"],
)
def test_validate_dependency_allows(dep):
    validate_dependency(dep)


@pytest.mark.parametrize("dep", ["--user", "-r requirements.txt", "torch && rm -rf /"])
def test_validate_dependency_blocks(dep):
    with pytest.raises(ValueError):
        validate_dependency(dep)


def test_validate_path_within_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    inside = root / "child"
    inside.touch()
    validate_path_within_root(inside, root)
    outside = tmp_path / "outside"
    outside.touch()
    with pytest.raises(ValueError):
        validate_path_within_root(outside, root)


def test_environment_context_restores():
    key = "PYISOLATE_TEST_ENV"
    original = os.environ.get(key)
    with environment(**{key: "123"}):
        assert os.environ.get(key) == "123"
    assert os.environ.get(key) == original


def test_deduplication_filter_suppresses_duplicates():
    f = _DeduplicationFilter(timeout_seconds=60)
    record = logging.LogRecord("x", logging.INFO, "", 0, "hello", args=(), exc_info=None)
    assert f.filter(record) is True
    record2 = logging.LogRecord("x", logging.INFO, "", 0, "hello", args=(), exc_info=None)
    assert f.filter(record2) is False


@pytest.mark.parametrize("platform,expected", [("linux", str), ("win32", False), ("darwin", False)])
def test_probe_cuda_ipc_support_non_linux(monkeypatch, platform, expected):
    monkeypatch.setattr(sys, "platform", platform)
    if platform == "linux":
        # simulate torch missing
        monkeypatch.setitem(sys.modules, "torch", None)
    supported, reason = _probe_cuda_ipc_support()
    if platform != "linux":
        assert supported is False
    else:
        assert supported is False


def test_ensure_uv_installed(monkeypatch, tmp_path):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/uv")
    ext = make_extension(tmp_path)
    assert ext._ensure_uv() is True


def test_ensure_uv_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("shutil.which", lambda _: None)
    called = {}

    def fake_check_call(*args, **kwargs):
        called["invoked"] = True
        raise RuntimeError("fail")

    monkeypatch.setattr("subprocess.check_call", fake_check_call)
    ext = make_extension(tmp_path)
    assert ext._ensure_uv() is False
    assert called.get("invoked") is True


def test_exclude_satisfied_requirements_skips_torch(monkeypatch, tmp_path):
    ext = make_extension(tmp_path, share_torch=True)

    def fake_run(cmd, capture_output, text, check):
        return SimpleNamespace(stdout=json.dumps([
            {"name": "torch", "version": "2.2.0"},
            {"name": "torchvision", "version": "0.17.0"},
            {"name": "pillow", "version": "10.0.0"},
        ]))

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("pyisolate._internal.host.get_torch_ecosystem_packages", lambda: {"torch", "torchvision"})
    deps = ["torch==2.2.0", "torchvision==0.17.0", "pillow==10.0.0"]
    filtered = ext._exclude_satisfied_requirements(deps, Path("/tmp/python"))
    # pillow already installed matching spec, should be skipped as well
    assert filtered == []


def test_extension_ensure_process_started_no_double_init(monkeypatch, tmp_path):
    ext = make_extension(tmp_path)
    called = {"count": 0}
    monkeypatch.setattr(ext, "_initialize_process", lambda: called.__setitem__("count", called["count"] + 1))
    ext.ensure_process_started()
    ext.ensure_process_started()
    assert called["count"] == 1


def test_extension_get_proxy_memoized(monkeypatch, tmp_path):
    ext = make_extension(tmp_path)
    ext.rpc = MagicMock()
    proxy_obj = object()
    ext.rpc.create_caller.return_value = proxy_obj
    p1 = ext.get_proxy()
    p2 = ext.get_proxy()
    assert p1 is p2


def test_extension_getattr_forwards(monkeypatch, tmp_path):
    ext = make_extension(tmp_path)
    ext.rpc = MagicMock()
    ext.extension_type = SimpleNamespace
    proxy_obj = SimpleNamespace(run=lambda: "ok")
    ext.rpc.create_caller.return_value = proxy_obj
    monkeypatch.setattr(ext, "ensure_process_started", lambda: None)
    assert ext.run() == "ok"