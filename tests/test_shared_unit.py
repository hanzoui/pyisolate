import asyncio
import os
import queue
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pyisolate._internal import shared
from pyisolate._internal.shared import (
    AttrDict,
    AttributeContainer,
    LocalMethodRegistry,
    ProxiedSingleton,
    AsyncRPC,
    _prepare_for_rpc,
    _tensor_to_cuda,
)


@pytest.fixture(autouse=True)
def reset_env(monkeypatch):
    monkeypatch.delenv("PYISOLATE_CHILD", raising=False)
    monkeypatch.delenv("PYISOLATE_ENABLE_CUDA_IPC", raising=False)
    shared._cuda_ipc_env_enabled = False
    shared._ipc_metrics["send_cuda_ipc"] = 0
    shared._ipc_metrics["send_cuda_fallback"] = 0


@pytest.fixture
def dummy_torch(monkeypatch):
    class Tensor:
        def __init__(self, is_cuda=False):
            self.is_cuda = is_cuda
            self.cpu_called = False

        def cpu(self):
            self.cpu_called = True
            return self

    dummy = SimpleNamespace(Tensor=Tensor)
    monkeypatch.setitem(sys.modules, "torch", dummy)
    return dummy


@pytest.fixture
def comfy_proxies(monkeypatch):
    class ModelPatcherProxy:
        def __init__(self, model_id, registry=None, manage_lifecycle=False):
            self.model_id = model_id

    class ModelPatcherRegistry:
        def __init__(self):
            self._inst = {}

        def register(self, obj):
            self._inst[1] = obj
            return 1

        def _get_instance(self, model_id):
            return self._inst.get(model_id, f"mp-{model_id}")

    class ModelSamplingProxy:
        def __init__(self, ms_id):
            self.ms_id = ms_id

    class ModelSamplingRegistry:
        def __init__(self):
            self._inst = {}

        def register(self, obj):
            self._inst[1] = obj
            return 1

        def _get_instance(self, ms_id):
            return self._inst.get(ms_id, f"ms-{ms_id}")

    monkeypatch.setitem(sys.modules, "comfy.isolation.model_patcher_proxy", SimpleNamespace(ModelPatcherProxy=ModelPatcherProxy, ModelPatcherRegistry=ModelPatcherRegistry))
    monkeypatch.setitem(sys.modules, "comfy.isolation.model_sampling_proxy", SimpleNamespace(ModelSamplingProxy=ModelSamplingProxy, ModelSamplingRegistry=ModelSamplingRegistry))


def test_attrdict_and_attribute_container():
    a = AttrDict({"x": 1})
    assert a.x == 1
    b = a.copy()
    assert isinstance(b, AttrDict) and b.x == 1

    c = AttributeContainer({"y": 2})
    assert c.y == 2
    c2 = c.copy()
    assert c2.y == 2


def test_prepare_for_rpc_primitives(dummy_torch):
    assert _prepare_for_rpc(5) == 5
    t = dummy_torch.Tensor(is_cuda=False)
    assert _prepare_for_rpc(t) is t


def test_prepare_for_rpc_cuda_ipc(dummy_torch, monkeypatch):
    shared._cuda_ipc_env_enabled = True
    t = dummy_torch.Tensor(is_cuda=True)
    out = _prepare_for_rpc(t)
    assert out is t
    assert shared._ipc_metrics["send_cuda_ipc"] == 1


def test_prepare_for_rpc_cuda_fallback(dummy_torch):
    t = dummy_torch.Tensor(is_cuda=True)
    out = _prepare_for_rpc(t)
    assert out is t  # stub cpu returns self
    assert t.cpu_called is True
    assert shared._ipc_metrics["send_cuda_fallback"] == 1


def test_prepare_for_rpc_model_patcher_child_raises(comfy_proxies, monkeypatch):
    monkeypatch.setenv("PYISOLATE_CHILD", "1")
    with pytest.raises(RuntimeError):
        _prepare_for_rpc(type("ModelPatcher", (), {})())


def test_tensor_to_cuda_refs(comfy_proxies, monkeypatch):
    monkeypatch.setenv("PYISOLATE_CHILD", "1")
    ref = {"__type__": "ModelPatcherRef", "model_id": 1}
    obj = _tensor_to_cuda(ref)
    assert getattr(obj, "model_id", None) == 1

    ref_ms = {"__type__": "ModelSamplingRef", "ms_id": 2}
    obj2 = _tensor_to_cuda(ref_ms)
    assert getattr(obj2, "ms_id", None) == 2


def test_local_method_registry():
    class Demo(ProxiedSingleton):
        @shared.local_execution
        def ping(self):
            return "pong"

    registry = LocalMethodRegistry.get_instance()
    registry.register_class(Demo)
    assert registry.is_local_method(Demo, "ping") is True
    assert registry.get_local_method(Demo, "ping")() == "pong"


def test_proxied_singleton_use_remote(monkeypatch):
    class Demo(ProxiedSingleton):
        @classmethod
        def get_remote_id(cls):
            return "demo"

    rpc = MagicMock()
    proxy_obj = SimpleNamespace()
    rpc.create_caller.return_value = proxy_obj
    Demo.use_remote(rpc)
    assert Demo.get_instance() is proxy_obj


def test_async_rpc_dispatch_and_response(monkeypatch):
    # Use real multiprocessing queues for thread safety
    import multiprocessing

    recv_q = multiprocessing.get_context("spawn").Queue()
    send_q = multiprocessing.get_context("spawn").Queue()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    rpc = AsyncRPC(recv_queue=recv_q, send_queue=send_q)

    class Callee:
        async def add(self, x, y):
            return x + y

    rpc.register_callee(Callee(), "cid")

    # simulate incoming call
    recv_q.put({
        "kind": "call",
        "object_id": "cid",
        "call_id": 0,
        "parent_call_id": None,
        "method": "add",
        "args": (1, 2),
        "kwargs": {},
    })
    recv_q.put(None)

    rpc.run()
    loop.run_until_complete(rpc.run_until_stopped())

    resp = send_q.get(timeout=1)
    assert resp["kind"] == "response"
    assert resp["result"] == 3
    assert resp["error"] is None
    loop.close()


def test_async_rpc_register_callback_and_call(monkeypatch):
    import multiprocessing

    recv_q = multiprocessing.get_context("spawn").Queue()
    send_q = multiprocessing.get_context("spawn").Queue()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    rpc = AsyncRPC(recv_queue=recv_q, send_queue=send_q)

    async def cb(val):
        return val * 2

    cb_id = rpc.register_callback(cb)
    recv_q.put({
        "kind": "callback",
        "callback_id": cb_id,
        "call_id": 0,
        "parent_call_id": None,
        "args": (5,),
        "kwargs": {},
    })
    recv_q.put(None)

    rpc.run()
    loop.run_until_complete(rpc.run_until_stopped())

    resp = send_q.get(timeout=1)
    assert resp["kind"] == "response"
    assert resp["result"] == 10
    assert resp["error"] is None
    loop.close()
