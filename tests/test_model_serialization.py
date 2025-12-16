import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pyisolate._internal.model_serialization import (
    deserialize_from_isolation,
    deserialize_proxy_result,
    serialize_for_isolation,
)


@pytest.fixture(autouse=True)
def reset_env(monkeypatch):
    monkeypatch.delenv("PYISOLATE_ENABLE_CUDA_IPC", raising=False)
    monkeypatch.delenv("PYISOLATE_CHILD", raising=False)


@pytest.fixture
def dummy_comfy_modules(monkeypatch):
    # RemoteObjectHandle
    class RemoteObjectHandle:
        def __init__(self, object_id):
            self.object_id = object_id

    # ModelPatcher registry
    class ModelPatcherRegistry:
        def __init__(self):
            self._last = None

        def register(self, obj):
            self._last = obj
            return 123

        def _get_instance(self, model_id):
            return f"mp-{model_id}"

    class ModelPatcherProxy:
        def __init__(self, model_id, registry=None, manage_lifecycle=False):
            self.model_id = model_id
            self.registry = registry
            self.manage_lifecycle = manage_lifecycle

    # CLIP registry
    class CLIPRegistry:
        def register(self, obj):
            return 321

        def _get_instance(self, clip_id):
            return f"clip-{clip_id}"

    class CLIPProxy:
        def __init__(self, clip_id, registry=None, manage_lifecycle=False):
            self.clip_id = clip_id

    # VAE registry
    class VAERegistry:
        def register(self, obj):
            return 222

        def _get_instance(self, vae_id):
            return f"vae-{vae_id}"

    class VAEProxy:
        def __init__(self, vae_id):
            self.vae_id = vae_id

    # ModelSampling registry
    class ModelSamplingRegistry:
        def register(self, obj):
            return 555

        def _get_instance(self, ms_id):
            return f"ms-{ms_id}"

    class ModelSamplingProxy:
        def __init__(self, ms_id):
            self.ms_id = ms_id

    modules = {
        "comfy.isolation.extension_wrapper": SimpleNamespace(RemoteObjectHandle=RemoteObjectHandle),
        "comfy.isolation.model_patcher_proxy": SimpleNamespace(ModelPatcherRegistry=ModelPatcherRegistry, ModelPatcherProxy=ModelPatcherProxy),
        "comfy.isolation.clip_proxy": SimpleNamespace(CLIPRegistry=CLIPRegistry, CLIPProxy=CLIPProxy),
        "comfy.isolation.vae_proxy": SimpleNamespace(VAERegistry=VAERegistry, VAEProxy=VAEProxy),
        "comfy.isolation.model_sampling_proxy": SimpleNamespace(ModelSamplingRegistry=ModelSamplingRegistry, ModelSamplingProxy=ModelSamplingProxy),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return modules


class DummyTensor:
    def __init__(self, is_cuda):
        self.is_cuda = is_cuda
        self.cpu_called = False

    def cpu(self):
        self.cpu_called = True
        return "cpu-tensor"


@pytest.fixture
def patch_torch_tensor(monkeypatch):
    import torch

    monkeypatch.setattr(torch, "Tensor", DummyTensor)
    return torch


def test_serialize_basic_passthrough(dummy_comfy_modules):
    assert serialize_for_isolation(5) == 5
    assert serialize_for_isolation("hello") == "hello"
    data = {"a": 1, "b": [2, 3]}
    assert serialize_for_isolation(data) == data


def test_serialize_remote_object_handle(dummy_comfy_modules):
    handle_cls = dummy_comfy_modules["comfy.isolation.extension_wrapper"].RemoteObjectHandle
    obj = type("Obj", (), {})()
    obj._pyisolate_remote_handle = handle_cls(42)
    assert serialize_for_isolation(obj) is obj._pyisolate_remote_handle


def test_serialize_model_patcher_registers(dummy_comfy_modules):
    mp = type("ModelPatcher", (), {})()
    result = serialize_for_isolation(mp)
    assert result == {"__type__": "ModelPatcherRef", "model_id": 123}


def test_serialize_model_patcher_child_requires_instance_id(dummy_comfy_modules, monkeypatch):
    monkeypatch.setenv("PYISOLATE_CHILD", "1")
    mp = type("ModelPatcher", (), {})()
    with pytest.raises(RuntimeError):
        serialize_for_isolation(mp)


def test_serialize_clip_and_proxy(dummy_comfy_modules):
    clip = type("CLIP", (), {})()
    result = serialize_for_isolation(clip)
    assert result == {"__type__": "CLIPRef", "clip_id": 321}

    proxy = type("CLIPProxy", (), {"_instance_id": 99})()
    result_proxy = serialize_for_isolation(proxy)
    assert result_proxy == {"__type__": "CLIPRef", "clip_id": 99}


def test_serialize_vae_and_proxy(dummy_comfy_modules):
    vae = type("VAE", (), {})()
    result = serialize_for_isolation(vae)
    assert result == {"__type__": "VAERef", "vae_id": 222}

    proxy = type("VAEProxy", (), {"_instance_id": 77})()
    result_proxy = serialize_for_isolation(proxy)
    assert result_proxy == {"__type__": "VAERef", "vae_id": 77}


def test_serialize_model_sampling_and_proxy(dummy_comfy_modules, monkeypatch):
    ms = type("ModelSampling", (), {})()
    result = serialize_for_isolation(ms)
    assert result == {"__type__": "ModelSamplingRef", "ms_id": 555}

    proxy = type("ModelSamplingProxy", (), {"_instance_id": 42})()
    result_proxy = serialize_for_isolation(proxy)
    # type_name startswith ModelSampling so registry registers again
    assert result_proxy == {"__type__": "ModelSamplingRef", "ms_id": 555}

    monkeypatch.setenv("PYISOLATE_CHILD", "1")
    child_ms = type("ModelSampling", (), {"_instance_id": 9})()
    child_result = serialize_for_isolation(child_ms)
    assert child_result == {"__type__": "ModelSamplingRef", "ms_id": 9}

    bad_child_ms = type("ModelSampling", (), {})()
    with pytest.raises(RuntimeError):
        serialize_for_isolation(bad_child_ms)


def test_serialize_cuda_tensor_falls_back_to_cpu(monkeypatch, patch_torch_tensor):
    tensor = DummyTensor(is_cuda=True)
    monkeypatch.delenv("PYISOLATE_ENABLE_CUDA_IPC", raising=False)
    result = serialize_for_isolation(tensor)
    assert tensor.cpu_called is True
    assert result == "cpu-tensor"


def test_serialize_cuda_tensor_ipc_enabled(monkeypatch, patch_torch_tensor):
    tensor = DummyTensor(is_cuda=True)
    monkeypatch.setenv("PYISOLATE_ENABLE_CUDA_IPC", "1")
    import pyisolate._internal.model_serialization as ms

    monkeypatch.setattr(ms, "_cuda_ipc_enabled", True)
    result = serialize_for_isolation(tensor)
    # Should return original tensor, no CPU copy
    assert result is tensor
    assert tensor.cpu_called is False


def test_serialize_cpu_tensor_passthrough(monkeypatch, patch_torch_tensor):
    tensor = DummyTensor(is_cuda=False)
    result = serialize_for_isolation(tensor)
    assert result is tensor
    assert tensor.cpu_called is False


@pytest.mark.asyncio
async def test_deserialize_model_patcher_reference(dummy_comfy_modules):
    data = {"__type__": "ModelPatcherRef", "model_id": 999}
    result = await deserialize_from_isolation(data, extension=None)
    assert result == "mp-999"


@pytest.mark.asyncio
async def test_deserialize_remote_object_handle_resolves_when_extension(dummy_comfy_modules):
    handle_cls = dummy_comfy_modules["comfy.isolation.extension_wrapper"].RemoteObjectHandle
    handle = handle_cls(object_id=7)
    resolved = SimpleNamespace()

    class Ext:
        async def get_remote_object(self, object_id):
            assert object_id == 7
            return resolved

    result = await deserialize_from_isolation(handle, extension=Ext())
    assert result is resolved
    assert getattr(resolved, "_pyisolate_remote_handle", None) is handle


@pytest.mark.asyncio
async def test_deserialize_clip_reference(dummy_comfy_modules):
    data = {"__type__": "CLIPRef", "clip_id": 11}
    result = await deserialize_from_isolation(data, extension=None)
    assert result == "clip-11"


@pytest.mark.asyncio
async def test_deserialize_vae_reference(dummy_comfy_modules):
    data = {"__type__": "VAERef", "vae_id": 22}
    result = await deserialize_from_isolation(data, extension=None)
    assert result == "vae-22"


@pytest.mark.asyncio
async def test_deserialize_model_sampling_reference(dummy_comfy_modules):
    data = {"__type__": "ModelSamplingRef", "ms_id": 33}
    result = await deserialize_from_isolation(data, extension=None)
    assert result == "ms-33"


@pytest.mark.asyncio
async def test_deserialize_nested_structures(dummy_comfy_modules):
    nested = {
        "clip": {"__type__": "CLIPRef", "clip_id": 1},
        "vae": [{"__type__": "VAERef", "vae_id": 2}],
        "ms": ({"__type__": "ModelSamplingRef", "ms_id": 3},),
    }
    result = await deserialize_from_isolation(nested, extension=None)
    assert result == {"clip": "clip-1", "vae": ["vae-2"], "ms": ("ms-3",)}


def test_deserialize_proxy_result_clip(dummy_comfy_modules):
    data = {"__type__": "CLIPRef", "clip_id": 5}
    result = deserialize_proxy_result(data)
    assert result.clip_id == 5


def test_deserialize_proxy_result_vae(dummy_comfy_modules):
    data = {"__type__": "VAERef", "vae_id": 6}
    result = deserialize_proxy_result(data)
    assert result.vae_id == 6


def test_deserialize_proxy_result_model_sampling(dummy_comfy_modules):
    data = {"__type__": "ModelSamplingRef", "ms_id": 7}
    result = deserialize_proxy_result(data)
    assert result.ms_id == 7


@pytest.mark.asyncio
async def test_deserialize_registry_handler(monkeypatch):
    # Simulate registry handling custom type
    class FakeRegistry:
        def has_handler(self, key):
            return key == "CustomRef"

        def get_deserializer(self, key):
            assert key == "CustomRef"
            return lambda d: {"ok": d["value"]}

    monkeypatch.setattr("pyisolate._internal.model_serialization.SerializerRegistry.get_instance", lambda: FakeRegistry())
    data = {"__type__": "CustomRef", "value": 10}
    result = await deserialize_from_isolation(data, extension=None)
    assert result == {"ok": 10}


def test_deserialize_proxy_result_registry(monkeypatch):
    class FakeRegistry:
        def has_handler(self, key):
            return key == "CustomRef"

        def get_deserializer(self, key):
            assert key == "CustomRef"
            return lambda d: {"proxied": d["value"]}

    monkeypatch.setattr("pyisolate._internal.model_serialization.SerializerRegistry.get_instance", lambda: FakeRegistry())
    data = {"__type__": "CustomRef", "value": 9}
    result = deserialize_proxy_result(data)
    assert result == {"proxied": 9}


def test_deserialize_proxy_result_model_patcher(dummy_comfy_modules):
    data = {"__type__": "ModelPatcherRef", "model_id": 77}
    result = deserialize_proxy_result(data)
    assert result.model_id == 77
    assert result.manage_lifecycle is False
