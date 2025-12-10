"""
Custom serialization helpers for PyIsolate.

These helpers let PyIsolate transparently move tensors and registered objects
across process boundaries. ModelPatcher/CLIP/VAE objects are converted to
lightweight references while preserving tensor sharing semantics.
"""

import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)


def prepare_tensors_for_rpc(data: Any) -> Any:
    """Recursively move CUDA tensors to CPU shared memory for zero-copy transfer.

    This handles the tensor transport layer for host→child RPC calls. We prefer CPU
    shared memory for debuggability even though CUDA IPC is possible.
    """
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
    """Recursively move CPU tensors back to the specified device.

    Used on the host to move RPC arguments from CPU shared memory back to GPU
    before executing ModelPatcher or other host calls.
    """
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
    """Serialize data for transmission to an isolated process (host side).

    ModelPatcher/CLIP/VAE objects are converted to reference dictionaries so the
    isolated process can fetch them lazily. RemoteObjectHandle instances are passed
    through to preserve identity without pickling heavyweight objects.
    """
    type_name = type(data).__name__

    # If this object originated as a RemoteObjectHandle, prefer to send the
    # handle back to the isolated process rather than attempting to pickle the
    # concrete instance. This preserves identity (and avoids pickling large or
    # unpicklable objects) while still allowing host-side consumers to interact
    # with the resolved object.
    try:
        from comfy.isolation.extension_wrapper import RemoteObjectHandle

        handle = getattr(data, "_pyisolate_remote_handle", None)
        if isinstance(handle, RemoteObjectHandle):
            return handle
    except Exception:
        # If the helper cannot be imported or attribute access fails, continue
        # with normal serialization.
        pass

    if type_name == 'ModelPatcher':
        logger.warning("[serialize_for_isolation] ModelPatcher child=%s has_id=%s", os.environ.get("PYISOLATE_CHILD") == "1", hasattr(data, "_instance_id"))
        if os.environ.get("PYISOLATE_CHILD") == "1":
            if hasattr(data, "_instance_id"):
                return {"__type__": "ModelPatcherRef", "model_id": getattr(data, "_instance_id")}
            return {"__type__": "ModelPatcherOpaque", "repr": str(data)[:256]}
        from comfy.isolation.model_patcher_proxy import ModelPatcherRegistry
        model_id = ModelPatcherRegistry().register(data)
        return {"__type__": "ModelPatcherRef", "model_id": model_id}
    
    if type_name == 'ModelPatcherProxy':
        # Already a proxy, return ref dict
        return {"__type__": "ModelPatcherRef", "model_id": data._instance_id}

    if type_name == 'CLIP':
        try:
            from comfy.isolation.clip_proxy import CLIPRegistry
            clip_id = CLIPRegistry().register(data)
            return {"__type__": "CLIPRef", "clip_id": clip_id}
        except ImportError:
            return data
    
    if type_name == 'CLIPProxy':
        # Already a proxy, return ref dict
        return {"__type__": "CLIPRef", "clip_id": data._instance_id}

    if type_name == 'VAE':
        try:
            from comfy.isolation.vae_proxy import VAERegistry
            vae_id = VAERegistry().register(data)
            return {"__type__": "VAERef", "vae_id": vae_id}
        except ImportError:
            return data
    
    if type_name == 'VAEProxy':
        # Already a proxy, return as-is (return the ref dict)
        return {"__type__": "VAERef", "vae_id": data._instance_id}

    if type_name.startswith('ModelSampling'):
        logger.warning("[serialize_for_isolation] ModelSampling child=%s has_id=%s", os.environ.get("PYISOLATE_CHILD") == "1", hasattr(data, "_instance_id"))
        if os.environ.get("PYISOLATE_CHILD") == "1":
            if hasattr(data, "_instance_id"):
                return {"__type__": "ModelSamplingRef", "ms_id": getattr(data, "_instance_id")}
            return {"__type__": "ModelSamplingOpaque", "repr": str(data)[:256]}
        try:
            import copyreg
            from comfy.isolation.model_sampling_proxy import ModelSamplingRegistry, ModelSamplingProxy

            def _reduce_model_sampling(ms):
                registry = ModelSamplingRegistry()
                ms_id_local = registry.register(ms)
                return (ModelSamplingProxy, (ms_id_local,))

            copyreg.pickle(type(data), _reduce_model_sampling)

            ms_id = ModelSamplingRegistry().register(data)
            return {"__type__": "ModelSamplingRef", "ms_id": ms_id}
        except ImportError:
            return data

    if type_name == 'ModelSamplingProxy':
        return {"__type__": "ModelSamplingRef", "ms_id": data._instance_id}

    if isinstance(data, dict):
        if data.get("__type__") == "ModelPatcherRef":
            return data
        return {k: serialize_for_isolation(v) for k, v in data.items()}

    if isinstance(data, (list, tuple)):
        result = [serialize_for_isolation(item) for item in data]
        return type(data)(result)

    return data


async def deserialize_from_isolation(data: Any, extension: Any = None, _nested: bool = False) -> Any:
    """Deserialize data received from an isolated process (host side).

    Top-level ``RemoteObjectHandle`` values are resolved to concrete objects when an
    extension proxy is available. Nested handles stay opaque so they can be returned
    back to the child without forcing unnecessary pickling/unpickling.
    """
    from comfy.isolation.extension_wrapper import RemoteObjectHandle

    type_name = type(data).__name__

    if isinstance(data, RemoteObjectHandle):
        if _nested or extension is None:
            return data
        try:
            resolved = await extension.get_remote_object(data.object_id)
            try:
                setattr(resolved, "_pyisolate_remote_handle", data)
            except Exception:
                pass
            return resolved
        except Exception:
            return data

    if type_name == 'NodeOutput':
        # Treat NodeOutput as a transparent container. Preserve current nesting
        # semantics so top-level outputs can be concretized while nested handles
        # stay opaque.
        return await deserialize_from_isolation(data.args, extension, _nested=_nested)

    if isinstance(data, dict):
        ref_type = data.get("__type__")

        if ref_type == "ModelPatcherRef":
            from comfy.isolation.model_patcher_proxy import ModelPatcherRegistry
            return ModelPatcherRegistry()._get_instance(data["model_id"])

        if ref_type == "CLIPRef":
            from comfy.isolation.clip_proxy import CLIPRegistry
            return CLIPRegistry()._get_instance(data["clip_id"])

        if ref_type == "VAERef":
            from comfy.isolation.vae_proxy import VAERegistry
            return VAERegistry()._get_instance(data["vae_id"])

        if ref_type == "ModelSamplingRef":
            from comfy.isolation.model_sampling_proxy import ModelSamplingRegistry
            return ModelSamplingRegistry()._get_instance(data["ms_id"])

        deserialized: dict[str, Any] = {}
        for k, v in data.items():
            # Dict entries are considered nested to preserve handles inside
            # structured payloads (e.g., da_model['model']).
            deserialized[k] = await deserialize_from_isolation(v, extension, _nested=True)
        return deserialized

    if isinstance(data, (list, tuple)):
        # For list/tuple, propagate the current nesting flag. Top-level tuples
        # (e.g., node outputs) stay `_nested=False`, allowing handles to resolve
        # to concrete objects when appropriate. Deeper levels inherit `_nested`
        # to avoid over-resolving nested handles.
        result = [await deserialize_from_isolation(item, extension, _nested=_nested) for item in data]
        return type(data)(result)

    return data


def deserialize_proxy_result(data: Any) -> Any:
    """Deserialize RPC results in the isolated process (child side).

    Reference dictionaries emitted by the host are converted into the appropriate
    proxy instances (ModelPatcherProxy, CLIPProxy, VAEProxy) while preserving
    container structure.
    """
    if isinstance(data, dict):
        ref_type = data.get("__type__")

        if ref_type == "ModelPatcherRef":
            from comfy.isolation.model_patcher_proxy import ModelPatcherProxy
            return ModelPatcherProxy(data["model_id"], registry=None, manage_lifecycle=False)

        if ref_type == "CLIPRef":
            from comfy.isolation.clip_proxy import CLIPProxy
            return CLIPProxy(data["clip_id"], registry=None, manage_lifecycle=False)

        if ref_type == "VAERef":
            from comfy.isolation.vae_proxy import VAEProxy
            return VAEProxy(data["vae_id"])

        if ref_type == "ModelSamplingRef":
            from comfy.isolation.model_sampling_proxy import ModelSamplingProxy
            return ModelSamplingProxy(data["ms_id"])

        return {k: deserialize_proxy_result(v) for k, v in data.items()}

    if isinstance(data, (list, tuple)):
        result = [deserialize_proxy_result(item) for item in data]
        return type(data)(result)

    return data
