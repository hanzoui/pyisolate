import os
import queue
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyisolate._internal import host
from pyisolate._internal.host import Extension


class DummyRPC:
    def __init__(self, *args, **kwargs):
        self.run_called = False

    def run(self):
        self.run_called = True


class DummyProcess:
    def __init__(self):
        self.alive = False

    def start(self):
        self.alive = True

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.alive = False

    def join(self, timeout=None):
        self.alive = False

    def kill(self):
        self.alive = False


class DummyContext:
    def __init__(self):
        self.q = queue.Queue()

    def Queue(self):
        return queue.Queue()

    def Process(self, target, args):
        return DummyProcess()


class DummyMP:
    def __init__(self):
        self.ctx = DummyContext()
        self.executable = None

    def get_context(self, mode):
        return self.ctx

    def set_executable(self, exe):
        self.executable = exe


class DummyExtension(Extension):
    def __init__(self, tmp_path: Path, config_overrides=None):
        base_config = {
            "name": "demo",
            "dependencies": [],
            "share_torch": True,
            "share_cuda_ipc": False,
            "apis": [],
        }
        if config_overrides:
            base_config.update(config_overrides)
        super().__init__(
            module_path="/tmp/mod.py",
            extension_type=SimpleNamespace,
            config=base_config,
            venv_root_path=str(tmp_path),
        )
        # patch multiprocessing
        self.mp = DummyMP()

    def _create_extension_venv(self):
        # skip actual venv creation
        return

    def _install_dependencies(self):
        return

    def __launch(self):
        return DummyProcess()


@pytest.fixture(autouse=True)
def reset_env(monkeypatch):
    monkeypatch.delenv("PYISOLATE_ENABLE_CUDA_IPC", raising=False)
    monkeypatch.delenv("PYISOLATE_CHILD", raising=False)


def test_initialize_process_requires_share_torch_for_cuda_ipc(tmp_path):
    ext = DummyExtension(tmp_path, {"share_torch": False, "share_cuda_ipc": True})
    with pytest.raises(RuntimeError):
        ext._initialize_process()


def test_initialize_process_cuda_ipc_unavailable_raises(monkeypatch, tmp_path):
    ext = DummyExtension(tmp_path, {"share_torch": True, "share_cuda_ipc": True})
    monkeypatch.setattr(host, "_probe_cuda_ipc_support", lambda: (False, "no"))
    with pytest.raises(RuntimeError):
        ext._initialize_process()


def test_initialize_process_sets_env_and_runs_rpc(monkeypatch, tmp_path):
    ext = DummyExtension(tmp_path, {"share_torch": True, "share_cuda_ipc": False})
    monkeypatch.setattr(host, "AsyncRPC", lambda recv_queue, send_queue: DummyRPC())
    ext._initialize_process()
    assert os.environ.get("PYISOLATE_ENABLE_CUDA_IPC") == "0"
    assert isinstance(ext.rpc, DummyRPC)
    assert ext.rpc.run_called is True


def test_install_dependencies_no_deps_returns(monkeypatch, tmp_path):
    ext = DummyExtension(tmp_path)
    # ensure python exe exists
    venv_bin = Path(ext.venv_path / "bin")
    venv_bin.mkdir(parents=True, exist_ok=True)
    exe = venv_bin / "python"
    exe.write_text("#!/usr/bin/env python")
    ext._install_dependencies()


def test_probe_cuda_ipc_support_handles_import_error(monkeypatch):
    monkeypatch.setattr(host.sys, "platform", "linux")
    monkeypatch.setitem(host.sys.modules, "torch", None)
    supported, reason = host._probe_cuda_ipc_support()
    assert supported is False
    assert "torch import failed" in reason


def test_install_dependencies_respects_lock_cache(monkeypatch, tmp_path):
    ext = DummyExtension(tmp_path)
    venv_bin = Path(ext.venv_path / "bin")
    venv_bin.mkdir(parents=True, exist_ok=True)
    exe = venv_bin / "python"
    exe.write_text("#!/usr/bin/env python")

    lock = ext.venv_path / ".pyisolate_deps.json"
    descriptor = {
        "dependencies": [],
        "share_torch": True,
        "torch_spec": None,
        "pyisolate": host.pyisolate_version,
        "python": host.sys.version,
    }
    import json, hashlib

    fp = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
    lock.write_text(json.dumps({"fingerprint": fp, "descriptor": descriptor}))

    # should return early without invoking pip/uv
    ext._install_dependencies()