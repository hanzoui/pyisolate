"""
pyisolate._internal.model_serialization
Custom serialization for ComfyUI ModelPatcher objects.

This module provides hooks for PyIsolate's RPC layer to automatically
convert ModelPatcher instances to proxies during cross-process communication.
"""

import torch
import logging
from typing import Any

logger = logging.getLogger(__name__)


def prepare_tensors_for_rpc(data: Any) -> Any:
    """Recursively move CUDA tensors to CPU shared memory for zero-copy transfer.
    
    This handles the tensor transport layer. Since we disabled cudaMallocAsync,
    we COULD use CUDA IPC, but for Phase 1-5 we use CPU shared memory as the
    safer, more debuggable path.
    
    Args:
        data: Arbitrary nested structure (list, dict, tensor, etc.)
        
    Returns:
        Same structure with CUDA tensors moved to CPU shared memory
    """
    if isinstance(data, torch.Tensor):
        if data.is_cuda:
            # Move to CPU and mark for shared memory
            cpu_tensor = data.to("cpu", copy=True)
            cpu_tensor.share_memory_()
            logger.debug(f"📚 [Serialization] Moved CUDA tensor to CPU shared memory (shape={data.shape})")
            return cpu_tensor
        elif not data.is_shared():
            # Already CPU, just mark for sharing
            data.share_memory_()
        return data
    
    elif isinstance(data, dict):
        return {k: prepare_tensors_for_rpc(v) for k, v in data.items()}
    
    elif isinstance(data, (list, tuple)):
        result = [prepare_tensors_for_rpc(item) for item in data]
        return type(data)(result)  # Preserve tuple vs list
    
    else:
        return data


def move_tensors_to_device(data: Any, device: torch.device) -> Any:
    """Recursively move CPU tensors back to specified device.
    
    Used on host-side to move RPC arguments from CPU shared memory
    back to GPU before executing ModelPatcher methods.
    
    Args:
        data: Arbitrary nested structure
        device: Target device (typically GPU)
        
    Returns:
        Same structure with tensors on target device
    """
    if isinstance(data, torch.Tensor):
        if str(data.device) != str(device):
            result = data.to(device)
            logger.debug(f"📚 [Serialization] Moved tensor to {device} (shape={data.shape})")
            return result
        return data
    
    elif isinstance(data, dict):
        return {k: move_tensors_to_device(v, device) for k, v in data.items()}
    
    elif isinstance(data, (list, tuple)):
        result = [move_tensors_to_device(item, device) for item in data]
        return type(data)(result)
    
    else:
        return data


def serialize_for_isolation(data: Any) -> Any:
    """Serialize data for transmission to isolated process.
    
    This is called on the HOST side before sending data to child.
    ModelPatcher instances are replaced with {"__type__": "ModelPatcherRef", "model_id": "..."}.
    CLIP instances are replaced with {"__type__": "CLIPRef", "clip_id": "..."}.
    
    Args:
        data: Arbitrary data (may contain ModelPatcher or CLIP)
        
    Returns:
        Serialized structure safe for cross-process transport
    """
    type_name = type(data).__name__
    
    # Check if it's a ModelPatcher (avoid importing to prevent circular deps)
    if type_name == 'ModelPatcher':
        from comfy.isolation.model_patcher_proxy import ModelPatcherRegistry
        model_registry = ModelPatcherRegistry()
        model_id = model_registry.register(data)
        logger.debug(f"📚 [Serialization] ModelPatcher → ModelPatcherRef({model_id})")
        return {
            "__type__": "ModelPatcherRef",
            "model_id": model_id,
        }
    
    # Check if it's a CLIP object
    elif type_name == 'CLIP':
        try:
            from comfy.isolation.clip_proxy import CLIPRegistry
            clip_registry = CLIPRegistry()
            clip_id = clip_registry.register(data)
            logger.debug(f"📚 [Serialization] CLIP → CLIPRef({clip_id})")
            return {
                "__type__": "CLIPRef",
                "clip_id": clip_id,
            }
        except ImportError:
            logger.warning("📚 [Serialization] ComfyUI integration not available")
            return data
    
    elif isinstance(data, dict):
        # Avoid double-wrapping refs
        if data.get("__type__") == "ModelPatcherRef":
            return data
        return {k: serialize_for_isolation(v) for k, v in data.items()}
    
    elif isinstance(data, (list, tuple)):
        result = [serialize_for_isolation(item) for item in data]
        return type(data)(result)
    
    else:
        # Primitives, tensors, etc. pass through
        return data


def deserialize_from_isolation(data: Any) -> Any:
    """Deserialize data received from isolated process.
    
    This is called on the HOST side after receiving results from child.
    ModelPatcherRef markers are resolved back to real ModelPatcher objects.
    CLIPRef markers are resolved back to real CLIP objects.
    
    Args:
        data: Serialized structure from child
        
    Returns:
        Deserialized structure with real ModelPatcher/CLIP instances
    """
    if isinstance(data, dict):
        ref_type = data.get("__type__")
        
        if ref_type == "ModelPatcherRef":
            from comfy.isolation.model_patcher_proxy import ModelPatcherRegistry
            model_registry = ModelPatcherRegistry()
            model_id = data["model_id"]
            patcher = model_registry._get_instance(model_id)
            logger.debug(f"📚 [Serialization] ModelPatcherRef({model_id}) → ModelPatcher")
            return patcher
        
        elif ref_type == "CLIPRef":
            clip_id = data["clip_id"]
            try:
                from comfy.isolation.clip_proxy import CLIPRegistry
                clip_registry = CLIPRegistry()
                clip = clip_registry._get_instance(clip_id)
                logger.debug(f"📚 [Serialization] CLIPRef({clip_id}) → CLIP")
                return clip
            except Exception as e:
                raise RuntimeError(
                    f"CLIP ID {clip_id} not found: {e}. "
                    f"Either the execution scope expired or the ID is invalid."
                )
        
        else:
            # Regular dict without __type__ - recurse into values
            return {k: deserialize_from_isolation(v) for k, v in data.items()}
    
    elif isinstance(data, (list, tuple)):
        result = [deserialize_from_isolation(item) for item in data]
        return type(data)(result)
    
    else:
        return data


def deserialize_proxy_result(data: Any) -> Any:
    """Deserialize result from RPC call (in isolated process).
    
    This is called on the CHILD side after receiving results from host.
    ModelPatcherRef markers are converted to ModelPatcherProxy instances.
    CLIPRef markers are converted to CLIPProxy instances.
    
    Args:
        data: Result from host RPC call
        
    Returns:
        Deserialized structure with proxy instances
    """
    if isinstance(data, dict):
        ref_type = data.get("__type__")
        
        if ref_type == "ModelPatcherRef":
            from comfy.isolation.model_patcher_proxy import ModelPatcherProxy
            model_id = data["model_id"]
            logger.debug(f"📚 [Serialization] ModelPatcherRef({model_id}) → ModelPatcherProxy")
            return ModelPatcherProxy(model_id, registry=None, manage_lifecycle=False)
        
        elif ref_type == "CLIPRef":
            from comfy.isolation.clip_proxy import CLIPProxy
            clip_id = data["clip_id"]
            logger.debug(f"📚 [Serialization] CLIPRef({clip_id}) → CLIPProxy")
            return CLIPProxy(clip_id, registry=None, manage_lifecycle=False)
        
        else:
            return {k: deserialize_proxy_result(v) for k, v in data.items()}
    
    elif isinstance(data, (list, tuple)):
        result = [deserialize_proxy_result(item) for item in data]
        return type(data)(result)
    
    else:
        return data
