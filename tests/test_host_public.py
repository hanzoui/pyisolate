import types
from typing import Any

import pytest

from pyisolate.host import ExtensionManager


class FakeExtension:
    @classmethod
    def __class_getitem__(cls, item):
        return cls

    def __init__(self, module_path: str, extension_type: Any, config: dict[str, Any], venv_root_path: str) -> None:
        self.module_path = module_path
        self.extension_type = extension_type
        self.config = config
        self.venv_root_path = venv_root_path
        self.started = 0
        self.proxy_obj = types.SimpleNamespace(run=lambda: "ok")
        self.rpc = object()
        self.stopped = 0

    def ensure_process_started(self) -> None:
        self.started += 1

    def get_proxy(self):
        return self.proxy_obj

    def stop(self):
        self.stopped += 1


@pytest.fixture(autouse=True)
def patch_extension(monkeypatch):
    monkeypatch.setattr("pyisolate.host.Extension", FakeExtension)


def make_manager(tmp_path):
    return ExtensionManager(types.SimpleNamespace, {"venv_root_path": str(tmp_path)})


def base_config(tmp_path):
    return {
        "name": "demo",
        "module_path": "/tmp/mod.py",
        "dependencies": [],
        "share_torch": True,
        "share_cuda_ipc": False,
        "apis": [],
        "venv_root_path": str(tmp_path),
    }


def test_load_extension_returns_host_extension(monkeypatch, tmp_path):
    mgr = make_manager(tmp_path)
    proxy = mgr.load_extension(base_config(tmp_path))
    # First access triggers start + rpc init + proxy creation
    assert proxy.proxy.run() == "ok"
    assert getattr(proxy, "_rpc", None) is mgr.extensions["demo"].rpc
    # Subsequent access uses cached proxy, no extra starts
    proxy.proxy
    ext = mgr.extensions["demo"]
    assert isinstance(ext, FakeExtension)
    assert ext.started == 1


def test_duplicate_extension_name_raises(tmp_path):
    mgr = make_manager(tmp_path)
    cfg = base_config(tmp_path)
    mgr.load_extension(cfg)
    with pytest.raises(ValueError):
        mgr.load_extension(cfg)


def test_host_extension_getattr_delegates(monkeypatch, tmp_path):
    mgr = make_manager(tmp_path)
    cfg = base_config(tmp_path)
    proxy = mgr.load_extension(cfg)
    # Add attr to underlying extension
    ext = mgr.extensions["demo"]
    ext.special = "hello"
    assert proxy.special == "hello"
    # Attribute missing on extension should delegate to proxy
    assert proxy.run() == "ok"


def test_stop_all_extensions_calls_stop(tmp_path):
    mgr = make_manager(tmp_path)
    cfg = base_config(tmp_path)
    mgr.load_extension(cfg)
    mgr.load_extension({**cfg, "name": "demo2"})
    mgr.stop_all_extensions()
    assert mgr.extensions == {}


def test_stop_all_extensions_logs_error(caplog, tmp_path):
    mgr = make_manager(tmp_path)
    cfg = base_config(tmp_path)
    proxy = mgr.load_extension(cfg)
    ext = mgr.extensions["demo"]

    def boom():
        raise RuntimeError("boom")

    ext.stop = boom  # type: ignore[assignment]

    with caplog.at_level("ERROR"):
        mgr.stop_all_extensions()

    assert "Error stopping extension 'demo'" in caplog.text
    assert mgr.extensions == {}