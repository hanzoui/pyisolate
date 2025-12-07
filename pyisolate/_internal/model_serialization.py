import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def prepare_tensors_for_rpc(data: Any) -> Any:
    """Move CUDA tensors to CPU shared memory for zero-copy transfer."""
    if isinstance(data, torch.Tensor):
        if data.is_cuda:
            cpu_tensor = data.to("cpu", copy=True)
            cpu_tensor.share_memory_()
            return cpu_tensor
        elif not data.is_shared():
            data.share_memory_()
        return data

    if isinstance(data, dict):
        return {k: prepare_tensors_for_rpc(v) for k, v in data.items()}

    if isinstance(data, (list, tuple)):
        result = [prepare_tensors_for_rpc(item) for item in data]
        return type(data)(result)

    return data


def move_tensors_to_device(data: Any, device: torch.device) -> Any:
    """Move CPU tensors to specified device."""
    if isinstance(data, torch.Tensor):
        if str(data.device) != str(device):
            return data.to(device)
        return data

    if isinstance(data, dict):
        return {k: move_tensors_to_device(v, device) for k, v in data.items()}

    if isinstance(data, (list, tuple)):
        result = [move_tensors_to_device(item, device) for item in data]
        return type(data)(result)

    return data


def serialize_for_isolation(data: Any) -> Any:
    """Serialize data for transmission to isolated process (host side)."""
    type_name = type(data).__name__

    if type_name == 'ModelPatcher':
        from comfy.isolation.model_patcher_proxy import ModelPatcherRegistry
        model_id = ModelPatcherRegistry().register(data)
        return {"__type__": "ModelPatcherRef", "model_id": model_id}

    if type_name == 'CLIP':
        try:
            from comfy.isolation.clip_proxy import CLIPRegistry
            clip_id = CLIPRegistry().register(data)
            return {"__type__": "CLIPRef", "clip_id": clip_id}
        except ImportError:
            return data

    if isinstance(data, dict):
        if data.get("__type__") == "ModelPatcherRef":
            return data
        return {k: serialize_for_isolation(v) for k, v in data.items()}

    if isinstance(data, (list, tuple)):
        result = [serialize_for_isolation(item) for item in data]
        return type(data)(result)

    return data


def deserialize_from_isolation(data: Any) -> Any:
    """Deserialize data from isolated process (host side)."""
    type_name = type(data).__name__
    if type_name == 'NodeOutput':
        return deserialize_from_isolation(data.args)

    if isinstance(data, dict):
        ref_type = data.get("__type__")

        if ref_type == "ModelPatcherRef":
            from comfy.isolation.model_patcher_proxy import ModelPatcherRegistry
            return ModelPatcherRegistry()._get_instance(data["model_id"])

        if ref_type == "CLIPRef":
            from comfy.isolation.clip_proxy import CLIPRegistry
            return CLIPRegistry()._get_instance(data["clip_id"])

        return {k: deserialize_from_isolation(v) for k, v in data.items()}

    if isinstance(data, (list, tuple)):
        result = [deserialize_from_isolation(item) for item in data]
        return type(data)(result)

    return data


def deserialize_proxy_result(data: Any) -> Any:
    """Deserialize RPC result in isolated process (child side)."""
    if isinstance(data, dict):
        ref_type = data.get("__type__")

        if ref_type == "ModelPatcherRef":
            from comfy.isolation.model_patcher_proxy import ModelPatcherProxy
            return ModelPatcherProxy(data["model_id"], registry=None, manage_lifecycle=False)

        if ref_type == "CLIPRef":
            from comfy.isolation.clip_proxy import CLIPProxy
            return CLIPProxy(data["clip_id"], registry=None, manage_lifecycle=False)

        return {k: deserialize_proxy_result(v) for k, v in data.items()}

    if isinstance(data, (list, tuple)):
        result = [deserialize_proxy_result(item) for item in data]
        return type(data)(result)

    return data
