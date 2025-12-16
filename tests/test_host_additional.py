import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pyisolate._internal import host
from pyisolate._internal.host import Extension, _probe_cuda_ipc_support
from pyisolate.config import ExtensionConfig


def make_extension(tmp_path: Path, *, share_torch: bool = True, share_cuda_ipc: bool = False, deps: list[str] | None = None) -> Extension:
    config: ExtensionConfig = {
        "name": "demo",
        "dependencies": deps or [],
        "share_torch": share_torch,
        "share_cuda_ipc": share_cuda_ipc,
        "apis": [],
    }
    return Extension(
        module_path=str(tmp_path / "mod"),
        extension_type=SimpleNamespace,
        config=config,
        venv_root_path=str(tmp_path / "venvs"),
    )


def test_detect_pyisolate_version_fallback(monkeypatch):
    monkeypatch.setattr(host.importlib_metadata, "version", lambda name: (_ for _ in ()).throw(Exception("oops")))
    assert host._detect_pyisolate_version() == "0.0.0"


def test_probe_cuda_ipc_support_success(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")

    class DummyEvent:
        def __init__(self, interprocess: bool = False) -> None:
            self.interprocess = interprocess

    class DummyCuda:
        def is_available(self):
            return True

        def current_device(self):
            return 0

        def Event(self, interprocess=False):  # noqa: N802  # mimic torch API
            return DummyEvent(interprocess=interprocess)

    class DummyTorch:
        def __init__(self):
            self.cuda = DummyCuda()

        def empty(self, *_, **kwargs):
            assert kwargs.get("device") == "cuda"
            return "tensor"

    monkeypatch.setitem(sys.modules, "torch", DummyTorch())
    supported, reason = _probe_cuda_ipc_support()
    assert supported is True
    assert reason == "ok"


def test_probe_cuda_ipc_support_cuda_false(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")

    class DummyCuda:
        def is_available(self):
            return False

    class DummyTorch:
        cuda = DummyCuda()

    monkeypatch.setitem(sys.modules, "torch", DummyTorch())
    supported, reason = _probe_cuda_ipc_support()
    assert supported is False
    assert "available" in reason


def test_probe_cuda_ipc_support_event_failure(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")

    class DummyCuda:
        def is_available(self):
            return True

        def current_device(self):
            return 0

        def Event(self, interprocess=False):  # noqa: N802
            raise RuntimeError("bad event")

    class DummyTorch:
        def __init__(self):
            self.cuda = DummyCuda()

        def empty(self, *_, **__):
            return "tensor"

    monkeypatch.setitem(sys.modules, "torch", DummyTorch())
    supported, reason = _probe_cuda_ipc_support()
    assert supported is False
    assert "failed" in reason


def test_initialize_process_requires_share_torch_for_cuda(monkeypatch, tmp_path):
    ext = make_extension(tmp_path, share_torch=False, share_cuda_ipc=True)
    with pytest.raises(RuntimeError):
        ext._initialize_process()


def test_initialize_process_probe_failure(monkeypatch, tmp_path):
    ext = make_extension(tmp_path, share_torch=True, share_cuda_ipc=True)
    monkeypatch.setattr(host, "_probe_cuda_ipc_support", lambda: (False, "nope"))
    with pytest.raises(RuntimeError):
        ext._initialize_process()


def test_initialize_process_success_sets_env_and_registers(monkeypatch, tmp_path):
    ext = make_extension(tmp_path, share_torch=False, share_cuda_ipc=False)

    class DummyCtx:
        def Queue(self):  # noqa: N802  # mimics multiprocessing ctx
            import multiprocessing

            return multiprocessing.Queue()

        def Process(self, target, args):  # noqa: N802
            class DummyProc:
                def start(self):
                    return None

                def is_alive(self):
                    return False

                def terminate(self):
                    return None

                def join(self, timeout=None):
                    return None

                def kill(self):
                    return None

            self.last_target = target
            self.last_args = args
            return DummyProc()

    class DummyMP:
        def get_context(self, mode):
            return DummyCtx()

        def set_executable(self, *_):
            return None

    class DummyListener:
        def __init__(self, *_):
            self.stopped = False

        def start(self):
            return None

        def stop(self):
            self.stopped = True

    class DummyRPC:
        def __init__(self, recv_queue, send_queue):
            self.recv = recv_queue
            self.send = send_queue
            self.ran = False
            self.registered = []

        def create_caller(self, *_):
            return SimpleNamespace()

        def run(self):
            self.ran = True

    class DummyAPI:
        def __init__(self):
            self.registered = False

        def _register(self, rpc):
            self.registered = True

    ext.config["apis"] = [DummyAPI]
    monkeypatch.setattr(ext, "mp", DummyMP())
    monkeypatch.setattr(host.logging, "StreamHandler", lambda *_: SimpleNamespace(addFilter=lambda *_: None))
    monkeypatch.setattr(host, "QueueListener", DummyListener)
    monkeypatch.setattr(host, "AsyncRPC", DummyRPC)
    monkeypatch.setattr(ext, "_Extension__launch", lambda: SimpleNamespace(is_alive=lambda: False))

    ext._initialize_process()

    assert os.environ.get("PYISOLATE_ENABLE_CUDA_IPC") == "0"
    assert isinstance(ext.rpc, DummyRPC)
    assert ext.rpc.ran is True


def test_extension_stop_collects_errors(monkeypatch, tmp_path):
    ext = make_extension(tmp_path)
    ext.proc = SimpleNamespace(
        is_alive=lambda: True,
        terminate=lambda: (_ for _ in ()).throw(RuntimeError("term")),
        join=lambda timeout=None: None,
        kill=lambda: None,
    )
    ext.log_listener = SimpleNamespace(stop=lambda: (_ for _ in ()).throw(RuntimeError("log")))
    ext.to_extension = SimpleNamespace(close=lambda: (_ for _ in ()).throw(RuntimeError("to")))
    ext.from_extension = SimpleNamespace(close=lambda: (_ for _ in ()).throw(RuntimeError("from")))
    ext.log_queue = SimpleNamespace(close=lambda: (_ for _ in ()).throw(RuntimeError("logq")))
    ext.manager = SimpleNamespace(shutdown=lambda: (_ for _ in ()).throw(RuntimeError("mgr")))

    with pytest.raises(RuntimeError) as exc:
        ext.stop()

    message = str(exc.value)
    assert "terminate" in message and "log_listener" in message and "manager" in message


def test_install_dependencies_skips_when_fingerprint_matches(monkeypatch, tmp_path):
    ext = make_extension(tmp_path, deps=["pkg==1.0"])
    python_exe = ext.venv_path / "bin" / "python"
    python_exe.parent.mkdir(parents=True, exist_ok=True)
    python_exe.write_text("#!/bin/python")

    monkeypatch.setattr(ext, "_ensure_uv", lambda: False)
    monkeypatch.setattr(ext, "_exclude_satisfied_requirements", lambda deps, exe: deps)

    descriptor = {
        "dependencies": ["pkg==1.0"],
        "share_torch": True,
        "torch_spec": None,
        "pyisolate": host.pyisolate_version,
        "python": sys.version,
    }
    fingerprint = host.hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
    lock_path = ext.venv_path / ".pyisolate_deps.json"
    ext.venv_path.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps({"fingerprint": fingerprint, "descriptor": descriptor}))

    ext._install_dependencies()
    # Should not remove or modify lock_path when skipping
    assert lock_path.exists()


def test_install_dependencies_uv_hardlink_fallback(monkeypatch, tmp_path):
    ext = make_extension(tmp_path, deps=["pkg==1.0"], share_torch=False)
    python_exe = ext.venv_path / "bin" / "python"
    python_exe.parent.mkdir(parents=True, exist_ok=True)
    python_exe.write_text("#!/bin/python")

    monkeypatch.setattr(ext, "_ensure_uv", lambda: True)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(host, "os", host.os)

    def fake_link(src, dst):
        raise OSError("no hardlink")

    monkeypatch.setattr(host.os, "link", fake_link)

    class FakeProc:
        def __init__(self):
            self.stdout = ["installing", "done"]

        def wait(self):
            return 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

    def fake_popen(cmd, stdout, stderr, text, bufsize):  # noqa: ARG001
        return FakeProc()

    monkeypatch.setattr(host.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(ext, "_exclude_satisfied_requirements", lambda deps, exe: deps)

    ext._install_dependencies()
    lock_path = ext.venv_path / ".pyisolate_deps.json"
    assert lock_path.exists()


def test_install_dependencies_failure_raises(monkeypatch, tmp_path):
    ext = make_extension(tmp_path, deps=["pkg==1.0"])
    python_exe = ext.venv_path / "bin" / "python"
    python_exe.parent.mkdir(parents=True, exist_ok=True)
    python_exe.write_text("#!/bin/python")

    monkeypatch.setattr(ext, "_ensure_uv", lambda: False)
    monkeypatch.setattr(ext, "_exclude_satisfied_requirements", lambda deps, exe: deps)

    class FakeProc:
        def __init__(self):
            self.stdout = ["bad"]

        def wait(self):
            return 1

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

    def fake_popen(cmd, stdout, stderr, text, bufsize):  # noqa: ARG001
        return FakeProc()

    monkeypatch.setattr(host.subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError):
        ext._install_dependencies()


def test_install_dependencies_missing_python_exe(monkeypatch, tmp_path):
    ext = make_extension(tmp_path, deps=["pkg==1.0"])
    with pytest.raises(RuntimeError):
        ext._install_dependencies()


def test_create_extension_venv_writes_parent_pth(monkeypatch, tmp_path):
    ext = make_extension(tmp_path, share_torch=True)

    def fake_check_call(cmd):  # noqa: ARG001
        ext.venv_path.mkdir(parents=True, exist_ok=True)
        site_dir = ext.venv_path / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        site_dir.mkdir(parents=True, exist_ok=True)

    host_site = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    host_site.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(host.subprocess, "check_call", lambda *args, **kwargs: fake_check_call(args))
    monkeypatch.setattr(host.site, "getsitepackages", lambda: [str(host_site)])

    ext._create_extension_venv()
    pth_file = list(ext.venv_path.rglob("_pyisolate_parent.pth"))
    assert pth_file, "Expected parent pth file to be created"
    assert host_site.as_posix() in pth_file[0].read_text()


def test_create_extension_venv_no_parent_site(monkeypatch, tmp_path):
    ext = make_extension(tmp_path, share_torch=True)

    def fake_check_call(cmd):  # noqa: ARG001
        ext.venv_path.mkdir(parents=True, exist_ok=True)
        site_dir = ext.venv_path / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        site_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(host.subprocess, "check_call", lambda *args, **kwargs: fake_check_call(args))
    monkeypatch.setattr(host.site, "getsitepackages", lambda: [])
    monkeypatch.setattr(sys, "prefix", "/nonexistent/prefix")

    with pytest.raises(RuntimeError):
        ext._create_extension_venv()


def test_original_launch_path_uses_environment(monkeypatch, tmp_path):
    ext = make_extension(tmp_path, share_torch=False)

    created = {}
    monkeypatch.setattr(ext, "_create_extension_venv", lambda: created.setdefault("venv", True))
    monkeypatch.setattr(ext, "_install_dependencies", lambda: created.setdefault("deps", True))

    class DummyProc:
        def __init__(self):
            self.started = False

        def start(self):
            self.started = True

        def is_alive(self):
            return self.started

        def terminate(self):
            self.started = False

        def join(self, timeout=None):
            self.started = False

        def kill(self):
            self.started = False

    class DummyCtx:
        def __init__(self):
            self.proc = DummyProc()

        def Process(self, target, args):  # noqa: N802
            return self.proc

    ext.ctx = DummyCtx()
    ext.mp = SimpleNamespace(set_executable=lambda *_, **__: None)

    # restore original __launch for this test
    monkeypatch.setattr(host.Extension, "_Extension__launch", host.Extension._orig_launch)

    proc = ext._Extension__launch()  # type: ignore[attr-defined]
    assert created == {"venv": True, "deps": True}
    assert proc.started is True
