"""
Custom serialization helpers for PyIsolate.

These helpers let PyIsolate transparently move tensors and registered objects
across process boundaries. ModelPatcher/CLIP/VAE objects are converted to
lightweight references while preserving tensor sharing semantics (CUDA tensors
stay on-device when CUDA IPC is enabled; otherwise they fall back to CPU shared
memory for transport).
"""

import contextlib
import logging
import os
import sys
from typing import TYPE_CHECKING, Any

import torch

from .serialization_registry import SerializerRegistry

_cuda_ipc_enabled = sys.platform == "linux" and os.environ.get("PYISOLATE_ENABLE_CUDA_IPC") == "1"

if TYPE_CHECKING:  # pragma: no cover - typing aids
    pass  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


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
    except Exception as exc:
        # If the helper cannot be imported or attribute access fails, continue
        # with normal serialization.
        logger.debug("Remote handle check failed: %s", exc)

    # Adapter-registered serializers take precedence over built-in handlers
    registry = SerializerRegistry.get_instance()
    if registry.has_handler(type_name):
        serializer = registry.get_serializer(type_name)
        if serializer:
            return serializer(data)

    if isinstance(data, torch.Tensor):
        if data.is_cuda:
            if _cuda_ipc_enabled:
                return data
            return data.cpu()
        return data

    if type_name == 'ModelPatcher':
        if os.environ.get("PYISOLATE_CHILD") == "1":
            if hasattr(data, "_instance_id"):
                return {"__type__": "ModelPatcherRef", "model_id": data._instance_id}
            raise RuntimeError(
                f"ModelPatcher in child lacks _instance_id: "
                f"{type(data).__module__}.{type(data).__name__}"
            )
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
        if os.environ.get("PYISOLATE_CHILD") == "1":
            if hasattr(data, "_instance_id"):
                return {"__type__": "ModelSamplingRef", "ms_id": data._instance_id}
            raise RuntimeError(
                f"ModelSampling in child lacks _instance_id: "
                f"{type(data).__module__}.{type(data).__name__}"
            )
        try:
            import copyreg

            from comfy.isolation.model_sampling_proxy import ModelSamplingProxy, ModelSamplingRegistry

            def _reduce_model_sampling(ms: Any) -> tuple[type[ModelSamplingProxy], tuple[int]]:
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

    registry = SerializerRegistry.get_instance()

    if isinstance(data, RemoteObjectHandle):
        if _nested or extension is None:
            return data
        try:
            resolved = await extension.get_remote_object(data.object_id)
            with contextlib.suppress(Exception):
                resolved._pyisolate_remote_handle = data
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

        # Adapter-registered deserializers for reference dicts
        if ref_type and registry.has_handler(ref_type):
            deserializer = registry.get_deserializer(ref_type)
            if deserializer:
                return deserializer(data)

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

        # Adapter-registered deserializers for proxy-bound references
        registry = SerializerRegistry.get_instance()
        if ref_type and registry.has_handler(ref_type):
            deserializer = registry.get_deserializer(ref_type)
            if deserializer:
                return deserializer(data)

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
