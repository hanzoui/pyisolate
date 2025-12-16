import asyncio
import logging
import queue
import sys
from types import SimpleNamespace

import pytest

from pyisolate._internal import shared
from pyisolate._internal.shared import (
    AsyncRPC,
    ProxiedSingleton,
    RPCPendingRequest,
    RPCResponse,
    _prepare_for_rpc,
)
from pyisolate.shared import ExtensionBase


def test_prepare_for_rpc_model_patcher_registers(monkeypatch):
    monkeypatch.delenv("PYISOLATE_CHILD", raising=False)
    class DummyRegistry:
        def __init__(self):
            self.registered = []

        def register(self, obj):
            self.registered.append(obj)
            return 99

    monkeypatch.setitem(sys.modules, "comfy.isolation.model_patcher_proxy", SimpleNamespace(ModelPatcherRegistry=DummyRegistry))
    obj = type("ModelPatcher", (), {})()
    ref = _prepare_for_rpc(obj)
    assert ref == {"__type__": "ModelPatcherRef", "model_id": 99}


@pytest.mark.parametrize("typename", ["ModelPatcher", "ModelSamplingFoo"])
def test_prepare_for_rpc_child_missing_instance_id(monkeypatch, typename):
    monkeypatch.setenv("PYISOLATE_CHILD", "1")
    cls = type(typename, (), {})
    with pytest.raises(RuntimeError):
        _prepare_for_rpc(cls())


def test_prepare_for_rpc_model_patcher_proxy_missing(monkeypatch):
    obj = type("ModelPatcherProxy", (), {})()
    with pytest.raises(RuntimeError):
        _prepare_for_rpc(obj)


def test_prepare_for_rpc_iterable_converts_tuple(monkeypatch):
    res = _prepare_for_rpc((1, {"x": 2}))
    assert res == (1, {"x": 2})


def test_prepare_for_rpc_model_sampling_host(monkeypatch):
    monkeypatch.delenv("PYISOLATE_CHILD", raising=False)

    class DummyMSRegistry:
        def register(self, obj):
            return 5

    monkeypatch.setitem(sys.modules, "comfy.isolation.model_sampling_proxy", SimpleNamespace(ModelSamplingRegistry=DummyMSRegistry))
    obj = type("ModelSamplingFoo", (), {})()
    ref = _prepare_for_rpc(obj)
    assert ref == {"__type__": "ModelSamplingRef", "ms_id": 5}


def test_prepare_for_rpc_model_patcher_proxy_ok():
    obj = type("ModelPatcherProxy", (), {"_instance_id": 7})()
    ref = _prepare_for_rpc(obj)
    assert ref == {"__type__": "ModelPatcherRef", "model_id": 7}


def test_debugprint_enabled(monkeypatch, caplog):
    monkeypatch.setenv("PYISOLATE_DEBUG_RPC", "1")
    original = shared.debug_all_messages
    shared.debug_all_messages = True
    shared.logger.setLevel(logging.DEBUG)
    with caplog.at_level("DEBUG", logger=shared.logger.name):
        shared.debugprint("hello", "world")
    assert any("hello world" in rec.message for rec in caplog.records)
    shared.debug_all_messages = original


def test_set_get_child_rpc_instance():
    rpc = object()
    shared.set_child_rpc_instance(rpc)  # type: ignore[arg-type]
    assert shared.get_child_rpc_instance() is rpc
    shared.set_child_rpc_instance(None)


@pytest.mark.asyncio
async def test_dispatch_request_unknown_kind_sets_error():
    send_q: queue.Queue = queue.Queue()
    recv_q: queue.Queue = queue.Queue()
    rpc = AsyncRPC(recv_queue=recv_q, send_queue=send_q)
    req = {"kind": "weird", "call_id": 1, "object_id": "x", "method": "m", "args": (), "kwargs": {}, "parent_call_id": None}
    await rpc.dispatch_request(req)  # type: ignore[arg-type]
    msg = send_q.get_nowait()
    assert msg["kind"] == "response"
    assert msg["error"] is not None


@pytest.mark.asyncio
async def test_dispatch_request_callback_async():
    send_q: queue.Queue = queue.Queue()
    recv_q: queue.Queue = queue.Queue()
    rpc = AsyncRPC(recv_queue=recv_q, send_queue=send_q)

    async def cb(val):
        return val + 1

    cb_id = rpc.register_callback(cb)
    req = {
        "kind": "callback",
        "callback_id": cb_id,
        "call_id": 2,
        "parent_call_id": None,
        "args": (4,),
        "kwargs": {},
    }
    await rpc.dispatch_request(req)  # type: ignore[arg-type]
    resp = send_q.get_nowait()
    assert resp["result"] == 5 and resp["error"] is None


def test_send_thread_response_branch():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    send_q: queue.Queue = queue.Queue()
    recv_q: queue.Queue = queue.Queue()
    rpc = AsyncRPC(recv_queue=recv_q, send_queue=send_q)
    resp = RPCResponse(kind="response", call_id=0, result="ok", error=None)
    rpc.outbox.put(resp)  # type: ignore[arg-type]
    rpc.outbox.put(None)
    rpc._send_thread()
    out = send_q.get_nowait()
    assert out == resp
    loop.close()


def test_recv_thread_sets_exception_on_error_response():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    send_q: queue.Queue = queue.Queue()
    recv_q: queue.Queue = queue.Queue()
    rpc = AsyncRPC(recv_queue=recv_q, send_queue=send_q)
    fut = loop.create_future()
    rpc.pending[0] = RPCPendingRequest(
        kind="call",
        object_id="obj",
        parent_call_id=None,
        calling_loop=loop,
        future=fut,
        method="noop",
        args=(),
        kwargs={},
    )
    rpc.blocking_future = loop.create_future()
    recv_q.put({"kind": "response", "call_id": 0, "result": None, "error": "boom"})
    recv_q.put(None)
    rpc._recv_thread()
    loop.run_until_complete(asyncio.sleep(0))
    assert fut.done() is True
    with pytest.raises(Exception):
        fut.result()
    loop.close()


def test_use_remote_injects_type_hints():
    class Inner(ProxiedSingleton):
        pass

    class Outer(ProxiedSingleton):
        inner: Inner

    called = {}

    class FakeRPC:
        def create_caller(self, cls, object_id):  # noqa: ANN001
            called[(cls, object_id)] = True
            return SimpleNamespace(object_id=object_id)

    rpc = FakeRPC()
    Outer.use_remote(rpc)  # type: ignore[arg-type]
    outer = Outer.get_instance()
    assert getattr(outer, "inner").object_id == Inner.get_remote_id()
    shared.SingletonMetaclass._instances.clear()


def test_extension_base_helpers():
    class DummyRPC:
        def __init__(self):
            self.registered = []
            self.calls = []

        def register_callee(self, obj, oid):
            self.registered.append((obj, oid))

        def create_caller(self, obj_type, oid):
            self.calls.append((obj_type, oid))
            return {"oid": oid}

        async def stop(self):
            self.stopped = True

    class DemoExt(ExtensionBase):
        pass

    ext = DemoExt()
    rpc = DummyRPC()
    ext._initialize_rpc(rpc)  # type: ignore[arg-type]
    ext.register_callee(object(), "abc")
    caller = ext.create_caller(DemoExt, "xyz")
    assert caller == {"oid": "xyz"}
    assert rpc.registered and rpc.calls

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(ext.stop())
    loop.close()
    assert getattr(rpc, "stopped", False) is True
