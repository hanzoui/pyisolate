import asyncio
import sys
from types import SimpleNamespace

import pytest

from pyisolate._internal import model_serialization
from pyisolate._internal.serialization_registry import SerializerRegistry


@pytest.fixture(autouse=True)
def clear_registry():
    reg = SerializerRegistry.get_instance()
    reg.clear()
    yield
    reg.clear()


def test_serialize_remote_handle_passthrough(monkeypatch):
    class RemoteObjectHandle:
        pass

    monkeypatch.setitem(sys.modules, "comfy.isolation.extension_wrapper", SimpleNamespace(RemoteObjectHandle=RemoteObjectHandle))
    handle = RemoteObjectHandle()
    data = SimpleNamespace(_pyisolate_remote_handle=handle)
    assert model_serialization.serialize_for_isolation(data) is handle


def test_serialize_registry_handler(monkeypatch):
    reg = SerializerRegistry.get_instance()
    reg.register("Custom", lambda obj: {"s": obj.value})

    class Custom:
        def __init__(self, value):
            self.value = value

    obj = Custom(5)
    assert model_serialization.serialize_for_isolation(obj) == {"s": 5}


def test_serialize_model_patcher_host(monkeypatch):
    class DummyRegistry:
        def register(self, obj):
            return 11

    monkeypatch.delenv("PYISOLATE_CHILD", raising=False)
    monkeypatch.setitem(sys.modules, "comfy.isolation.model_patcher_proxy", SimpleNamespace(ModelPatcherRegistry=DummyRegistry))
    obj = type("ModelPatcher", (), {})()
    ref = model_serialization.serialize_for_isolation(obj)
    assert ref == {"__type__": "ModelPatcherRef", "model_id": 11}


def test_serialize_model_patcher_child_missing_id(monkeypatch):
    monkeypatch.setenv("PYISOLATE_CHILD", "1")
    obj = type("ModelPatcher", (), {})()
    with pytest.raises(RuntimeError):
        model_serialization.serialize_for_isolation(obj)


def test_serialize_model_patcher_proxy():
    obj = type("ModelPatcherProxy", (), {"_instance_id": 4})()
    ref = model_serialization.serialize_for_isolation(obj)
    assert ref == {"__type__": "ModelPatcherRef", "model_id": 4}


def test_serialize_model_sampling_host(monkeypatch):
    class DummyMSRegistry:
        def register(self, obj):
            return 3

    monkeypatch.delenv("PYISOLATE_CHILD", raising=False)
    monkeypatch.setitem(sys.modules, "comfy.isolation.model_sampling_proxy", SimpleNamespace(ModelSamplingRegistry=DummyMSRegistry, ModelSamplingProxy=SimpleNamespace))
    obj = type("ModelSamplingX", (), {})()
    ref = model_serialization.serialize_for_isolation(obj)
    assert ref == {"__type__": "ModelSamplingRef", "ms_id": 3}


def test_serialize_clip_proxy_and_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "comfy.isolation.clip_proxy", SimpleNamespace(CLIPRegistry=lambda: SimpleNamespace(register=lambda obj: 5)))
    clip = type("CLIPProxy", (), {"_instance_id": 8})()
    assert model_serialization.serialize_for_isolation(clip) == {"__type__": "CLIPRef", "clip_id": 8}

    # ImportError branch returns data unchanged
    monkeypatch.setitem(sys.modules, "comfy.isolation.clip_proxy", None)
    clip_obj = type("CLIP", (), {})()
    assert model_serialization.serialize_for_isolation(clip_obj) is clip_obj


def test_deserialize_model_refs(monkeypatch):
    monkeypatch.setitem(sys.modules, "comfy.isolation.model_patcher_proxy", SimpleNamespace(ModelPatcherRegistry=lambda: SimpleNamespace(_get_instance=lambda mid: {"mp": mid})))
    monkeypatch.setitem(sys.modules, "comfy.isolation.clip_proxy", SimpleNamespace(CLIPRegistry=lambda: SimpleNamespace(_get_instance=lambda cid: {"clip": cid})))
    monkeypatch.setitem(sys.modules, "comfy.isolation.vae_proxy", SimpleNamespace(VAERegistry=lambda: SimpleNamespace(_get_instance=lambda vid: {"vae": vid})))
    monkeypatch.setitem(sys.modules, "comfy.isolation.model_sampling_proxy", SimpleNamespace(ModelSamplingRegistry=lambda: SimpleNamespace(_get_instance=lambda ms: {"ms": ms})))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(model_serialization.deserialize_from_isolation({"__type__": "ModelPatcherRef", "model_id": 2}))
    assert result == {"mp": 2}
    result = loop.run_until_complete(model_serialization.deserialize_from_isolation({"__type__": "CLIPRef", "clip_id": 9}))
    assert result == {"clip": 9}
    result = loop.run_until_complete(model_serialization.deserialize_from_isolation({"__type__": "VAERef", "vae_id": 7}))
    assert result == {"vae": 7}
    result = loop.run_until_complete(model_serialization.deserialize_from_isolation({"__type__": "ModelSamplingRef", "ms_id": 6}))
    assert result == {"ms": 6}
    loop.close()


def test_deserialize_remote_handle_resolution(monkeypatch):
    class RemoteObjectHandle:
        def __init__(self, object_id):
            self.object_id = object_id

    monkeypatch.setitem(sys.modules, "comfy.isolation.extension_wrapper", SimpleNamespace(RemoteObjectHandle=RemoteObjectHandle))

    class Ext:
        async def get_remote_object(self, oid):  # noqa: ANN001
            return {"obj": oid}

    handle = RemoteObjectHandle(10)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(model_serialization.deserialize_from_isolation(handle, Ext()))
    assert result == {"obj": 10}
    assert getattr(result, "_pyisolate_remote_handle", handle) == handle
    loop.close()
