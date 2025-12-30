import base64
from typing import Any

import torch
import torch.multiprocessing.reductions as reductions


def serialize_tensor(t: torch.Tensor) -> dict[str, Any]:
    """Serialize a tensor to JSON-compatible format using shared memory."""
    if t.is_cuda:
        return _serialize_cuda_tensor(t)
    return _serialize_cpu_tensor(t)


def _serialize_cpu_tensor(t: torch.Tensor) -> dict[str, Any]:
    """Serialize CPU tensor using file_system shared memory strategy."""
    if not t.is_shared():
        t.share_memory_()

    storage = t.untyped_storage()
    sfunc, sargs = reductions.reduce_storage(storage)

    if sfunc.__name__ == 'rebuild_storage_filename':
        # sargs: (cls, manager_path, storage_key, size)
        return {
            "__type__": "TensorRef",
            "device": "cpu",
            "strategy": "file_system",
            "manager_path": sargs[1].decode('utf-8'),
            "storage_key": sargs[2].decode('utf-8'),
            "storage_size": sargs[3],
            "dtype": str(t.dtype),
            "tensor_size": list(t.size()),
            "tensor_stride": list(t.stride()),
            "tensor_offset": t.storage_offset(),
            "requires_grad": t.requires_grad
        }
    elif sfunc.__name__ == 'rebuild_storage_fd':
        # Force file_system strategy for JSON-RPC compatibility
        torch.multiprocessing.set_sharing_strategy('file_system')
        t.share_memory_()
        return _serialize_cpu_tensor(t)
    else:
        raise RuntimeError(f"Unsupported storage reduction: {sfunc.__name__}")


def _serialize_cuda_tensor(t: torch.Tensor) -> dict[str, Any]:
    """Serialize CUDA tensor using CUDA IPC."""
    try:
        func, args = reductions.reduce_tensor(t)
    except RuntimeError as e:
        if "received from another process" in str(e):
            # This tensor was received via IPC and can't be re-shared.
            # This typically happens when a node returns an unmodified input tensor.
            # Clone is required but expensive for large tensors.
            tensor_size_mb = t.numel() * t.element_size() / (1024 * 1024)
            import logging
            logger = logging.getLogger(__name__)

            if tensor_size_mb > 100:  # 100MB threshold
                logger.warning(
                    "PERFORMANCE: Cloning large CUDA tensor (%.1fMB) received from another process. "
                    "Consider modifying the node to avoid returning unmodified input tensors.",
                    tensor_size_mb
                )
            else:
                logger.debug(
                    "Cloning CUDA tensor (%.2fMB) received from another process",
                    tensor_size_mb
                )

            t = t.clone()
            func, args = reductions.reduce_tensor(t)
        else:
            raise
    # args: (cls, size, stride, offset, storage_type, dtype, device_idx, handle, storage_size,
    #        storage_offset, requires_grad, ref_counter_handle, ref_counter_offset,
    #        event_handle, event_sync_required)
    return {
        "__type__": "TensorRef",
        "device": "cuda",
        "device_idx": args[6],  # int device index
        "tensor_size": list(args[1]),
        "tensor_stride": list(args[2]),
        "tensor_offset": args[3],
        "dtype": str(args[5]),
        "handle": base64.b64encode(args[7]).decode('ascii'),
        "storage_size": args[8],
        "storage_offset": args[9],
        "requires_grad": args[10],
        "ref_counter_handle": base64.b64encode(args[11]).decode('ascii'),
        "ref_counter_offset": args[12],
        "event_handle": base64.b64encode(args[13]).decode('ascii') if args[13] else None,
        "event_sync_required": args[14]
    }


def deserialize_tensor(data: dict[str, Any]) -> torch.Tensor:
    """Deserialize a tensor from TensorRef format."""
    # If this is already a tensor (e.g., passed through by shared memory), return as-is
    if isinstance(data, torch.Tensor):
        return data
    # All formats now use TensorRef
    return _deserialize_legacy_tensor(data)


def _convert_lists_to_tuples(obj: Any) -> Any:
    """Recursively convert lists to tuples (PyTorch requires tuples for size/stride)."""
    if isinstance(obj, list):
        return tuple(_convert_lists_to_tuples(item) for item in obj)
    if isinstance(obj, dict):
        return {k: _convert_lists_to_tuples(v) for k, v in obj.items()}
    return obj


def _deserialize_legacy_tensor(data: dict[str, Any]) -> torch.Tensor:
    """Handle legacy TensorRef format for backward compatibility."""
    device = data["device"]
    dtype_str = data["dtype"]
    dtype = getattr(torch, dtype_str.split(".")[-1])

    if device == "cpu":
        if data.get("strategy") != "file_system":
            raise RuntimeError(f"Unsupported CPU strategy: {data.get('strategy')}")

        manager_path = data["manager_path"].encode('utf-8')
        storage_key = data["storage_key"].encode('utf-8')
        storage_size = data["storage_size"]

        # Rebuild UntypedStorage (no dtype arg)
        rebuilt_storage = reductions.rebuild_storage_filename(
            torch.UntypedStorage, manager_path, storage_key, storage_size
        )

        # Wrap in TypedStorage (required by rebuild_tensor)
        typed_storage = torch.storage.TypedStorage(
            wrap_storage=rebuilt_storage, dtype=dtype, _internal=True
        )

        # Rebuild tensor using new signature: (cls, storage, metadata)
        # metadata is (offset, size, stride, requires_grad)
        metadata = (
            data["tensor_offset"],
            tuple(data["tensor_size"]),
            tuple(data["tensor_stride"]),
            data["requires_grad"]
        )
        cpu_tensor: torch.Tensor = reductions.rebuild_tensor(  # type: ignore[assignment]
            torch.Tensor,
            typed_storage,
            metadata
        )
        return cpu_tensor

    elif device == "cuda":
        handle = base64.b64decode(data["handle"])
        ref_counter_handle = base64.b64decode(data["ref_counter_handle"])
        event_handle = base64.b64decode(data["event_handle"]) if data["event_handle"] else None
        device_idx = data.get("device_idx", 0)  # int device index

        cuda_tensor: torch.Tensor = reductions.rebuild_cuda_tensor(  # type: ignore[assignment]
            torch.Tensor,
            tuple(data["tensor_size"]),
            tuple(data["tensor_stride"]),
            data["tensor_offset"],
            torch.storage.TypedStorage,
            dtype,
            device_idx,  # int device index, not torch.device
            handle,
            data["storage_size"],
            data["storage_offset"],
            data["requires_grad"],
            ref_counter_handle,
            data["ref_counter_offset"],
            event_handle,
            data["event_sync_required"]
        )
        return cuda_tensor

    raise RuntimeError(f"Unsupported device: {device}")


def register_tensor_serializer(registry: Any) -> None:
    # Register both "Tensor" (type name) and "torch.Tensor" (full name) just in case
    registry.register("Tensor", serialize_tensor, deserialize_tensor)
    registry.register("torch.Tensor", serialize_tensor, deserialize_tensor)
    # Also register TensorRef for deserialization
    registry.register("TensorRef", None, deserialize_tensor)
    # Register TorchReduction for recursive deserialization
    registry.register("TorchReduction", None, deserialize_tensor)

    # Register PyTorch atom types for recursive serialization
    def serialize_dtype(obj: Any) -> str:
        return str(obj)

    def deserialize_dtype(data: str) -> Any:
        import torch
        # Handle "torch.float32" -> torch.float32
        dtype_name = data.split(".")[-1]
        return getattr(torch, dtype_name)

    def serialize_device(obj: Any) -> str:
        return str(obj)

    def deserialize_device(data: str) -> Any:
        import torch
        return torch.device(data)

    def serialize_size(obj: Any) -> list:
        return list(obj)

    def deserialize_size(data: list) -> Any:
        import torch
        return torch.Size(data)

    registry.register("dtype", serialize_dtype, deserialize_dtype)
    registry.register("device", serialize_device, deserialize_device)
    registry.register("Size", serialize_size, deserialize_size)
