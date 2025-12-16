import asyncio
import builtins
from types import ModuleType
import importlib
import importlib.metadata as importlib_metadata
import json
import logging
import os
import queue
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.requirements import Requirement

import pyisolate._internal.bootstrap as bootstrap
import pyisolate._internal.client as client
import pyisolate._internal.host as host_internal
import pyisolate._internal.loader as loader_internal
import pyisolate._internal.model_serialization as model_ser
import pyisolate._internal.serialization_registry as ser_reg
import pyisolate._internal.shared as shared_internal
from pyisolate import shared as shared_public


class _StubAsyncRPC:
    def __init__(self, *_, **__):
        self.stopped = False

    def register_callee(self, *_args, **_kwargs):
        return None

    def create_caller(self, *_args, **_kwargs):
        return None

    def run(self):
        return None

    async def run_until_stopped(self):
        return None

    async def stop(self):
        self.stopped = True


def test_loader_discovery_paths(monkeypatch):
    # No adapters and name provided -> ValueError
    monkeypatch.setenv("PYISOLATE_ADAPTER_OVERRIDE", "")
    class EP:
        def __init__(self, name):
            self.name = name

        def load(self):
            class Dummy:
                pass

            return Dummy

    def fake_entry_points_empty():
        class Obj:
            def select(self, group=None):
                return []

        return Obj()

    def fake_entry_points_multi():
        class Obj:
            def select(self, group=None):
                return [EP("a"), EP("b")]

        return Obj()

    monkeypatch.setattr(importlib_metadata, "entry_points", fake_entry_points_empty)
    with pytest.raises(ValueError):
        loader_internal.load_adapter("missing")

    # Ambiguous auto-detect -> ValueError listing available
    monkeypatch.delenv("PYISOLATE_ADAPTER_OVERRIDE", raising=False)
    monkeypatch.setattr(importlib_metadata, "entry_points", fake_entry_points_multi)
    with pytest.raises(ValueError):
        loader_internal.load_adapter()


def test_model_serialization_remote_handle(monkeypatch):
    class RemoteHandle:
        pass

    dummy_handle = RemoteHandle()
    obj = type("Dummy", (), {})()
    obj._pyisolate_remote_handle = dummy_handle

    fake_mod = SimpleNamespace(RemoteObjectHandle=RemoteHandle)
    monkeypatch.setitem(sys.modules, "comfy.isolation.extension_wrapper", fake_mod)
    assert model_ser.serialize_for_isolation(obj) is dummy_handle


def test_model_serialization_registry_override(monkeypatch):
    registry = ser_reg.SerializerRegistry.get_instance()
    registry.clear()
    registry.register("MyType", lambda v: {"ok": v})
    instance = type("MyType", (), {})()
    result = model_ser.serialize_for_isolation(instance)
    assert result == {"ok": instance}


def test_model_serialization_import_error(monkeypatch):
    # Trigger import failure branch
    def fake_import(name, *args, **kwargs):
        if name == "comfy.isolation.extension_wrapper":
            raise ImportError("boom")
        return real_import(name, *args, **kwargs)

    real_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", fake_import)
    class Dummy:
        pass
    obj = Dummy()
    assert model_ser.serialize_for_isolation(obj) is obj


def test_model_serialization_modelpatcher_child_missing(monkeypatch):
    monkeypatch.setenv("PYISOLATE_CHILD", "1")
    class ModelPatcher:
        pass
    with pytest.raises(RuntimeError):
        model_ser.serialize_for_isolation(ModelPatcher())
    monkeypatch.delenv("PYISOLATE_CHILD", raising=False)


def test_model_serialization_vae_import_error(monkeypatch):
    class VAE:
        pass
    def fake_import(name, *args, **kwargs):
        if name.startswith("comfy.isolation.vae_proxy"):
            raise ImportError("boom")
        return real_import(name, *args, **kwargs)
    real_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert model_ser.serialize_for_isolation(VAE()) is not None


def test_model_serialization_modelsampling_paths(monkeypatch):
    # Child missing id raises
    monkeypatch.setenv("PYISOLATE_CHILD", "1")
    class ModelSamplingMissing:
        __name__ = "ModelSamplingMissing"
    with pytest.raises(RuntimeError):
        model_ser.serialize_for_isolation(ModelSamplingMissing())
    monkeypatch.delenv("PYISOLATE_CHILD", raising=False)

    # Non-child with registry present executes copyreg path
    class MSProxy:
        def __init__(self, _id):
            self.ms_id = _id

    fake_registry = SimpleNamespace(register=lambda obj: 5)
    fake_mod = SimpleNamespace(ModelSamplingRegistry=lambda: fake_registry, ModelSamplingProxy=MSProxy)
    monkeypatch.setitem(sys.modules, "comfy.isolation.model_sampling_proxy", fake_mod)
    result = model_ser.serialize_for_isolation(type("ModelSamplingY", (), {})())
    assert result == {"__type__": "ModelSamplingRef", "ms_id": 5}


@pytest.mark.asyncio
async def test_deserialize_from_isolation_handles(monkeypatch):
    # RemoteObjectHandle with no extension
    class RH:
        object_id = 1
    monkeypatch.setitem(sys.modules, "comfy.isolation.extension_wrapper", SimpleNamespace(RemoteObjectHandle=RH))
    rh = RH()
    assert await model_ser.deserialize_from_isolation(rh, None) is rh

    # RemoteObjectHandle with extension raising
    class Ext:
        async def get_remote_object(self, _):
            raise RuntimeError("fail")
    assert await model_ser.deserialize_from_isolation(rh, Ext()) is rh

    # NodeOutput passthrough
    class NodeOutput:
        def __init__(self):
            self.args = (1, 2)
    assert await model_ser.deserialize_from_isolation(NodeOutput(), None) == (1, 2)


def test_deserialize_proxy_result_paths(monkeypatch):
    # Unknown ref dict recursion path
    data = {"x": {"__type__": "unknown", "k": 1}}
    result = model_ser.deserialize_proxy_result(data)
    assert result == {"x": {"__type__": "unknown", "k": 1}}


def test_shared_attrdict_and_container():
    ad = shared_internal.AttrDict({"a": 1})
    assert ad.a == 1
    with pytest.raises(AttributeError):
        _ = ad.missing
    ac = shared_internal.AttributeContainer({"x": 5})
    assert ac.x == 5 and ac["x"] == 5 and len(ac) == 1 and list(ac.keys()) == ["x"]
    assert ac.get("y", 2) == 2
    assert "x" in ac
    with pytest.raises(AttributeError):
        del ac.__dict__["_data"]
        _ = ac.z


@pytest.mark.asyncio
async def test_shared_run_until_stopped_and_register_callee(monkeypatch):
    loop = asyncio.get_event_loop()
    rpc = shared_internal.AsyncRPC(queue.SimpleQueue(), queue.SimpleQueue())
    # register_callee duplicate
    rpc.register_callee(object(), "id")
    with pytest.raises(ValueError):
        rpc.register_callee(object(), "id")

    # run_until_stopped when blocking_future None triggers run()
    monkeypatch.setattr(rpc, "run", lambda: setattr(rpc, "blocking_future", loop.create_future()))
    rpc.blocking_future = None
    rpc.blocking_future = None
    task = asyncio.create_task(rpc.run_until_stopped())
    # set result to unblock
    await asyncio.sleep(0)
    rpc.blocking_future.set_result(None)
    await task


@pytest.mark.asyncio
async def test_shared_create_caller_errors():
    rpc = shared_internal.AsyncRPC(queue.SimpleQueue(), queue.SimpleQueue())
    caller = rpc.create_caller(type("T", (), {"foo": 1}), "id")
    with pytest.raises(AttributeError):
        caller.foo  # not callable

    caller2 = rpc.create_caller(type("T2", (), {"bar": lambda self=None: None}), "id")
    with pytest.raises(ValueError):
        await caller2.bar()


def test_loader_legacy_and_dup(monkeypatch):
    # Force legacy importlib_metadata path and duplicate match error
    dummy_ep = SimpleNamespace(name="dup", load=lambda: type("A", (), {}))

    class EPs:
        def select(self, group=None):
            return [dummy_ep, dummy_ep]

    fake_mod = SimpleNamespace(entry_points=lambda: EPs())
    monkeypatch.setitem(sys.modules, "importlib_metadata", fake_mod)
    monkeypatch.setattr(loader_internal, "sys", SimpleNamespace(version_info=(3, 9, 0)))
    with pytest.raises(ValueError):
        loader_internal.load_adapter("dup")


def test_host_requirements_exception(monkeypatch, tmp_path):
    ext = _make_extension(tmp_path, {"share_torch": True})

    class DummyResult:
        stdout = "[]"

    monkeypatch.setattr(host_internal.subprocess, "run", lambda *a, **k: DummyResult())
    monkeypatch.setattr(host_internal, "get_torch_ecosystem_packages", lambda: set())
    req_mod = SimpleNamespace(Requirement=lambda s: (_ for _ in ()).throw(ValueError("bad")))
    monkeypatch.setitem(sys.modules, "packaging.requirements", req_mod)
    res = ext._exclude_satisfied_requirements(["oops"], Path("/tmp/py"))
    assert res == ["oops"]


def test_host_stop_kill_branch(tmp_path):
    ext = _make_extension(tmp_path)

    class Proc:
        def is_alive(self):
            return True

        def terminate(self):
            return None

        def join(self, timeout=None):
            return None

        def kill(self):
            setattr(self, "killed", True)

    ext.proc = Proc()
    ext.log_listener = SimpleNamespace(stop=lambda: None)
    ext.to_extension = ext.from_extension = ext.log_queue = SimpleNamespace(close=lambda: None)
    ext.rpc = object()
    ext.stop()
    assert getattr(ext.proc, "killed", False)


def test_host_create_extension_venv_windows(monkeypatch, tmp_path):
    ext = _make_extension(tmp_path, {"share_torch": True})
    monkeypatch.setattr(host_internal.os, "name", "nt")
    def fake_check_call(cmd):
        (ext.venv_path / "fallback" / "site-packages").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(host_internal.subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(host_internal.site, "getsitepackages", lambda: ["/tmp/host_site/site-packages"])
    monkeypatch.setattr(host_internal.sys, "prefix", "/tmp/host_site")
    monkeypatch.setattr(host_internal.sys, "path", ["/tmp/host_site/site-packages"])
    ext._create_extension_venv()


def test_host_install_dependencies_windows_paths(monkeypatch, tmp_path):
    ext = _make_extension(tmp_path)
    ext.config["dependencies"] = ["dep"]
    monkeypatch.setattr(host_internal.os, "name", "nt")
    (ext.venv_path / "Scripts").mkdir(parents=True, exist_ok=True)
    python_exe = ext.venv_path / "Scripts" / "python.exe"
    python_exe.write_text("")
    monkeypatch.setattr(host_internal.Extension, "_ensure_uv", lambda self: True)
    monkeypatch.setattr(host_internal.shutil, "which", lambda *_: "/usr/bin/uv")
    monkeypatch.setattr(host_internal.os, "link", lambda *a, **k: None)
    class DummyPopen:
        def __init__(self, *a, **k):
            self.stdout = ["ok"]
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def wait(self):
            return 0
    monkeypatch.setattr(host_internal.subprocess, "Popen", DummyPopen)
    ext._install_dependencies()


def test_host_install_dependencies_torch_cpu(monkeypatch, tmp_path):
    ext = _make_extension(tmp_path, {"share_torch": False})
    ext.config["dependencies"] = ["dep"]
    torch_mod = ModuleType("torch")
    torch_mod.__version__ = "1.0+cpu"
    torch_mod.version = SimpleNamespace(cuda=None)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setattr(host_internal.Extension, "_ensure_uv", lambda self: False)
    (ext.venv_path / "bin").mkdir(parents=True, exist_ok=True)
    (ext.venv_path / "bin" / "python").write_text("")
    class DummyPopen2:
        def __init__(self, *a, **k):
            self.stdout = ["ok"]
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def wait(self):
            return 0
    monkeypatch.setattr(host_internal.subprocess, "Popen", DummyPopen2)
    ext._install_dependencies()


def test_host_install_dependencies_lockfile_bad(monkeypatch, tmp_path):
    ext = _make_extension(tmp_path)
    ext.config["dependencies"] = ["dep"]
    monkeypatch.setattr(host_internal.Extension, "_ensure_uv", lambda self: False)
    (ext.venv_path / "bin").mkdir(parents=True, exist_ok=True)
    (ext.venv_path / "bin" / "python").write_text("")
    lock = ext.venv_path / ".pyisolate_deps.json"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("not json")
    class DummyPopen3:
        def __init__(self, *a, **k):
            self.stdout = ["ok"]
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def wait(self):
            return 0
    monkeypatch.setattr(host_internal.subprocess, "Popen", DummyPopen3)
    ext._install_dependencies()


def test_host_join_calls_proc(tmp_path):
    ext = _make_extension(tmp_path)
    called = {}
    ext.proc = SimpleNamespace(join=lambda: called.setdefault("joined", True))
    ext.join()
    assert called.get("joined")



def test_model_serialization_tensor_cpu(monkeypatch):
    class FakeTensor:
        is_cuda = True

        def cpu(self):
            return "cpu_tensor"

    class FakeTorch:
        Tensor = FakeTensor

    monkeypatch.setattr(model_ser, "torch", SimpleNamespace(Tensor=FakeTensor))
    assert model_ser.serialize_for_isolation(FakeTensor()) == "cpu_tensor"


def test_model_serialization_modelpatcher_child(monkeypatch):
    class ModelPatcher:
        def __init__(self, ident=None):
            if ident is not None:
                self._instance_id = ident

    fake_registry = SimpleNamespace(register=lambda obj: 123)
    fake_mod = SimpleNamespace(ModelPatcherRegistry=lambda: fake_registry)
    monkeypatch.setitem(sys.modules, "comfy.isolation.model_patcher_proxy", fake_mod)
    monkeypatch.setenv("PYISOLATE_CHILD", "0")
    mp = ModelPatcher()
    assert model_ser.serialize_for_isolation(mp) == {"__type__": "ModelPatcherRef", "model_id": 123}


def test_serialization_registry_overwrite_warning(monkeypatch, caplog):
    reg = ser_reg.SerializerRegistry.get_instance()
    reg.clear()
    reg.register("A", lambda v: v)
    with caplog.at_level(logging.WARNING):
        reg.register("A", lambda v: v)
    assert "Overwriting" in caplog.text


def test_shared_attribute_container_roundtrip():
    data = {"a": 1, "b": 2}
    c = shared_internal.AttributeContainer(data)
    assert c.a == 1 and c["b"] == 2
    dumped = c.__getstate__()
    new_c = shared_internal.AttributeContainer({})
    new_c.__setstate__(dumped)
    assert new_c.b == 2




def test_apply_sys_path_prefers_module_env(monkeypatch, tmp_path):
    called = {}

    def fake_build(host_paths, extra_paths, preferred_root):
        called["preferred_root"] = preferred_root
        return ["/fake"]

    monkeypatch.setenv("PYISOLATE_MODULE_PATH", str(tmp_path / "mod" / "__init__.py"))
    monkeypatch.setattr(bootstrap, "build_child_sys_path", fake_build)
    bootstrap._apply_sys_path({"context_data": {}})
    assert called["preferred_root"] == str(tmp_path)


def test_bootstrap_child_no_adapter(monkeypatch):
    snapshot = {"sys_path": [], "context_data": {}, "adapter_name": None}
    monkeypatch.setenv("PYISOLATE_HOST_SNAPSHOT", json.dumps(snapshot))
    monkeypatch.setattr(bootstrap, "_apply_sys_path", lambda _: None)
    assert bootstrap.bootstrap_child() is None


def test_client_bootstrap_adapter_reload(monkeypatch):
    sentinel = object()
    monkeypatch.setenv("PYISOLATE_CHILD", "1")
    monkeypatch.setattr(bootstrap, "bootstrap_child", lambda: sentinel)

    reloaded = importlib.reload(client)
    assert getattr(reloaded, "_adapter") is sentinel


def test_client_logging_queue_branch(monkeypatch, tmp_path):
    monkeypatch.setenv("PYISOLATE_CHILD", "1")
    log_q: queue.SimpleQueue = queue.SimpleQueue()
    monkeypatch.setattr(client, "AsyncRPC", _StubAsyncRPC)
    monkeypatch.setattr(client, "set_child_rpc_instance", lambda rpc: None)
    shared_mod = importlib.import_module("pyisolate._internal.shared")
    stub_rpc = _StubAsyncRPC()
    monkeypatch.setattr(shared_mod, "get_child_rpc_instance", lambda: stub_rpc)

    class DummyExt(shared_public.ExtensionBase):
        async def before_module_loaded(self):
            return None

        async def on_module_loaded(self, module):
            shared_internal = importlib.import_module("pyisolate._internal.shared")
            await shared_internal.get_child_rpc_instance().stop()

    config = {"share_torch": False, "apis": [], "name": "dummy"}
    module_dir = tmp_path
    (module_dir / "__init__.py").write_text("")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            client.async_entrypoint(
                str(module_dir), DummyExt, config, queue.SimpleQueue(), queue.SimpleQueue(), log_q
            )
        )
    finally:
        loop.close()
        asyncio.set_event_loop(None)
    assert any(isinstance(h, logging.Handler) for h in logging.getLogger().handlers)


def test_entrypoint_invokes_asyncio_run(monkeypatch):
    called = {}

    def fake_run(coro):
        called["ran"] = True
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(coro)
        finally:
            loop.close()

    monkeypatch.setattr(asyncio, "run", fake_run)
    tmp_dir = Path(tempfile.mkdtemp())
    (tmp_dir / "__init__.py").write_text("")

    monkeypatch.setattr(client, "AsyncRPC", _StubAsyncRPC)

    client.entrypoint(
        str(tmp_dir),
        shared_public.ExtensionBase,
        {"share_torch": False, "apis": [], "name": "n"},
        queue.SimpleQueue(),
        queue.SimpleQueue(),
        queue.SimpleQueue(),
    )
    assert called["ran"]


def test_dedup_filter_purges(monkeypatch):
    filt = host_internal._DeduplicationFilter(timeout_seconds=1)
    now_seen = {str(i): 0 for i in range(1005)}
    filt.last_seen = now_seen
    record = logging.LogRecord("x", logging.INFO, __file__, 0, "unique message", None, None)
    assert filt.filter(record)
    assert len(filt.last_seen) <= 1000


def test_build_snapshot_adapter_failures(monkeypatch):
    monkeypatch.setattr(host_internal, "load_adapter", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    snap = host_internal.build_extension_snapshot("/tmp/mod")
    assert snap["adapter_name"] is None

    class FakeAdapter:
        identifier = "id"

        def get_path_config(self, *_args, **_kwargs):
            raise RuntimeError("fail")

    monkeypatch.setattr(host_internal, "load_adapter", lambda: FakeAdapter())
    snap = host_internal.build_extension_snapshot("/tmp/mod")
    assert snap["preferred_root"] is None


def test_validate_dependency_empty():
    host_internal.validate_dependency("")


def _make_extension(tmp_path, config_override=None):
    config = {
        "name": "ext",
        "dependencies": [],
        "apis": [],
        "share_torch": False,
    }
    if config_override:
        config.update(config_override)
    return host_internal.Extension(str(tmp_path), shared_public.ExtensionBase, config, str(tmp_path / "venvs"))


def test_share_cuda_ipc_force(monkeypatch, tmp_path):
    monkeypatch.setenv("PYISOLATE_FORCE_CUDA_IPC", "1")
    ext = _make_extension(tmp_path, {"share_cuda_ipc": False})
    assert ext.config["share_cuda_ipc"] is True


def test_exclude_requirements_paths_and_errors(monkeypatch, tmp_path):
    ext = _make_extension(tmp_path, {"share_torch": True})

    class DummyResult:
        stdout = "[]"

    monkeypatch.setattr(host_internal.subprocess, "run", lambda *a, **k: DummyResult())
    monkeypatch.setattr(host_internal, "get_torch_ecosystem_packages", lambda: set())

    real_req = Requirement

    def fake_req(val):
        if val == "bad":
            raise ValueError("bad")
        return real_req(val)

    monkeypatch.setattr(host_internal, "Requirement", fake_req, raising=False)
    result = ext._exclude_satisfied_requirements([
        "-e .", "/abs/path", "bad"
    ], Path("/tmp/python"))
    assert "-e ." in result and "/abs/path" in result and "bad" in result


def test_initialize_process_get_context_failure(monkeypatch, tmp_path):
    ext = _make_extension(tmp_path)
    ext.mp = SimpleNamespace(get_context=lambda *_: (_ for _ in ()).throw(ValueError("fail")))
    with pytest.raises(RuntimeError):
        ext._initialize_process()


def test_initialize_process_cuda_ipc_enabled(monkeypatch, tmp_path):
    dummy_ctx = SimpleNamespace(
        Queue=lambda: queue.SimpleQueue(),
        Process=lambda **kwargs: SimpleNamespace(start=lambda: None)
    )

    class DummyTorchMP:
        @staticmethod
        def get_context(mode):
            assert mode == "spawn"
            return dummy_ctx

    monkeypatch.setattr(host_internal, "_probe_cuda_ipc_support", lambda: (True, ""))
    monkeypatch.setattr(host_internal, "AsyncRPC", _StubAsyncRPC)
    ext = _make_extension(tmp_path, {"share_torch": True, "share_cuda_ipc": True})
    ext.mp = DummyTorchMP()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        ext._initialize_process()
        assert ext._cuda_ipc_enabled is True
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def test_initialize_process_windows_branch(monkeypatch, tmp_path):
    dummy_ctx = SimpleNamespace(
        Manager=lambda: SimpleNamespace(Queue=lambda: queue.SimpleQueue()),
        Process=lambda **kwargs: SimpleNamespace(start=lambda: None),
    )

    class DummyMP:
        @staticmethod
        def get_context(mode):
            return dummy_ctx

    ext = _make_extension(tmp_path)
    monkeypatch.setattr(host_internal.os, "name", "nt")
    dummy_mp = SimpleNamespace(get_context=lambda *_: dummy_ctx)
    monkeypatch.setitem(sys.modules, "multiprocessing", dummy_mp)
    monkeypatch.setattr(host_internal, "AsyncRPC", _StubAsyncRPC)
    ext.mp = DummyMP()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        ext._initialize_process()
        assert ext.log_queue is not None
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def test_extension_getattr_guard(tmp_path):
    ext = _make_extension(tmp_path)
    with pytest.raises(AttributeError):
        ext.__getattr__("name")


def test_extension_stop_with_alive_proc(monkeypatch, tmp_path):
    ext = _make_extension(tmp_path)
    monkeypatch.setattr(host_internal, "AsyncRPC", _StubAsyncRPC)

    class Proc:
        def __init__(self):
            self.stage = 0

        def is_alive(self):
            val = self.stage == 0
            self.stage += 1
            return val

        def terminate(self):
            return None

        def join(self, timeout=None):
            return None

        def kill(self):
            return None

    ext.proc = Proc()
    ext.log_listener = SimpleNamespace(stop=lambda: None)
    ext.to_extension = SimpleNamespace(close=lambda: None)
    ext.from_extension = SimpleNamespace(close=lambda: None)
    ext.log_queue = SimpleNamespace(close=lambda: None)
    ext.rpc = object()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        ext.stop()
    finally:
        loop.close()
        asyncio.set_event_loop(None)
    assert ext._process_initialized is False


def test_launch_windows_paths(monkeypatch, tmp_path):
    ext = _make_extension(tmp_path)
    monkeypatch.setattr(host_internal.os, "name", "nt")
    monkeypatch.setattr(ext, "_create_extension_venv", lambda: None)
    monkeypatch.setattr(ext, "_install_dependencies", lambda: None)
    monkeypatch.setattr(host_internal, "build_extension_snapshot", lambda *_: {})

    monkeypatch.setattr(host_internal.Extension, "_Extension__launch", host_internal.Extension._orig_launch)

    dummy_proc = SimpleNamespace(start=lambda: None)

    class DummyCtx:
        def Process(self, **_):
            return dummy_proc

    ext.ctx = DummyCtx()
    ext.to_extension = ext.from_extension = ext.log_queue = object()
    proc = ext._Extension__launch()
    assert proc is dummy_proc


def test_ensure_uv_install(monkeypatch):
    calls = {"which": []}

    def fake_which(name):
        calls["which"].append(name)
        return None if len(calls["which"]) == 1 else "/usr/bin/uv"

    monkeypatch.setattr(host_internal.shutil, "which", fake_which)
    monkeypatch.setattr(host_internal.subprocess, "check_call", lambda *a, **k: None)
    assert host_internal.Extension._ensure_uv(SimpleNamespace()) is True


def test_create_extension_venv_missing_site_packages(monkeypatch, tmp_path):
    ext = _make_extension(tmp_path, {"share_torch": True})
    monkeypatch.setattr(host_internal.subprocess, "check_call", lambda *a, **k: (ext.venv_path / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages").parent.mkdir(parents=True, exist_ok=True))
    with pytest.raises(RuntimeError):
        ext._create_extension_venv()


def test_install_dependencies_empty(monkeypatch, tmp_path):
    ext = _make_extension(tmp_path)
    ext.config["dependencies"] = []
    monkeypatch.setattr(host_internal.Extension, "_ensure_uv", lambda self: False)
    (ext.venv_path / "bin").mkdir(parents=True, exist_ok=True)
    (ext.venv_path / "bin" / "python").write_text("")
    ext._install_dependencies()


def test_uv_link_mode_copy(monkeypatch, tmp_path):
    ext = _make_extension(tmp_path)
    ext.config["dependencies"] = ["dep1"]
    monkeypatch.setattr(host_internal.Extension, "_ensure_uv", lambda self: True)
    monkeypatch.setattr(host_internal.shutil, "which", lambda *_: None)
    monkeypatch.setattr(host_internal.os, "link", lambda *a, **k: (_ for _ in ()).throw(OSError("no link")))
    
    class DummyPopen:
        def __init__(self, *_args, **_kwargs):
            self.stdout = ["ok"]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def wait(self):
            return 0

    monkeypatch.setattr(host_internal.subprocess, "Popen", DummyPopen)
    ext.venv_path.mkdir(parents=True, exist_ok=True)
    (ext.venv_path / "bin").mkdir(exist_ok=True)
    (ext.venv_path / "bin" / "python").write_text("")
    ext._install_dependencies()
