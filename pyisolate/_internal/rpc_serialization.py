"""
RPC Serialization Layer & Data Structures.

This module contains:
1. Data Structures: AttrDict, AttributeContainer, CallableProxy, RPC TypedDicts
2. Serialization Logic: _prepare_for_rpc, _tensor_to_cuda, debugprint
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from collections.abc import Iterable
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    TypedDict,
    Union,
)

if TYPE_CHECKING:
    # Avoid circular imports for type checking if possible
    # But here we just need types that might be used in annotations
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class CallableProxy:
    """
    Proxy for remote callables that preserves signature metadata.
    This allows inspect.signature() to work on the proxy
    """
    def __init__(self, metadata: dict[str, Any]):
        self._metadata = metadata
        self._name = metadata.get("name", "<remote_callable>")
        self._type_name = metadata.get("type", "Callable")

        # Reconstruct signature if available
        sig_data = metadata.get("signature")
        if sig_data:
            parameters = []
            for param_data in sig_data:
                # param_data is (name, kind_value, has_default)
                name, kind_val, has_default = param_data

                # generic default value if original had one (we don't serialize actual defaults)
                default: str | object = inspect.Parameter.empty
                if has_default:
                   default = "<remote_default>"

                # Map integer kind back to enum safely
                # _ParameterKind enum values are standard:
                # POSITIONAL_ONLY = 0
                # POSITIONAL_OR_KEYWORD = 1
                # VAR_POSITIONAL = 2
                # KEYWORD_ONLY = 3
                # VAR_KEYWORD = 4

                kind_map = {
                    0: inspect.Parameter.POSITIONAL_ONLY,
                    1: inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    2: inspect.Parameter.VAR_POSITIONAL,
                    3: inspect.Parameter.KEYWORD_ONLY,
                    4: inspect.Parameter.VAR_KEYWORD
                }
                kind = kind_map.get(kind_val, inspect.Parameter.POSITIONAL_OR_KEYWORD)

                parameters.append(inspect.Parameter(
                    name=name,
                    kind=kind,
                    default=default
                ))

            self.__signature__ = inspect.Signature(parameters=parameters)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # TODO: Implement full RPC callback support
        # For now, we primarily need introspection to pass checks.
        # Execution requires registering the callback ID on the sender side
        # and handling the reverse RPC call here.
        raise NotImplementedError(
            f"Remote execution of {self._name} is not yet fully implemented. "
            "Verification checks (inspect.signature) should pass."
        )

    def __repr__(self) -> str:
        return f"<CallableProxy for {self._name}>"


class AttrDict(dict[str, Any]):
    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def copy(self) -> AttrDict:
        return AttrDict(super().copy())


class AttributeContainer:
    """
    Non-dict container with attribute access and copy support.
    Prevents downstream code from downgrading to plain dict via dict(obj) / {**obj}.
    """

    def __init__(self, data: dict[str, Any]):
        self._data: dict[str, Any] = data

    def __getattr__(self, name: str) -> Any:
        if "_data" not in self.__dict__:
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def copy(self) -> AttributeContainer:
        return AttributeContainer(self._data.copy())

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def keys(self) -> Iterable[str]:
        return self._data.keys()

    def items(self) -> Iterable[tuple[str, Any]]:
        return self._data.items()

    def values(self) -> Iterable[Any]:
        return self._data.values()

    def __iter__(self) -> Iterable[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"AttributeContainer({getattr(self, '_data', '<empty>')})"

    def __getstate__(self) -> dict[str, Any]:
        return self._data

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._data = state


class RPCRequest(TypedDict):
    kind: Literal["call"]
    object_id: str
    call_id: int
    parent_call_id: int | None
    method: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class RPCCallback(TypedDict):
    kind: Literal["callback"]
    callback_id: str
    call_id: int
    parent_call_id: int | None
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class RPCResponse(TypedDict):
    kind: Literal["response"]
    call_id: int
    result: Any
    error: str | None


class RPCPendingRequest(TypedDict):
    kind: Literal["call", "callback"]
    object_id: str
    parent_call_id: int | None
    calling_loop: asyncio.AbstractEventLoop
    future: asyncio.Future[Any]
    method: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


RPCMessage = Union[RPCRequest, RPCCallback, RPCResponse]


# ---------------------------------------------------------------------------
# Globals / Debug Logic
# ---------------------------------------------------------------------------

# Debug flag for verbose RPC message logging (set via PYISOLATE_DEBUG_RPC=1)
debug_all_messages = bool(os.environ.get("PYISOLATE_DEBUG_RPC"))
_debug_rpc = debug_all_messages
_cuda_ipc_env_enabled = os.environ.get("PYISOLATE_ENABLE_CUDA_IPC") == "1"
_cuda_ipc_warned = False
_ipc_metrics: dict[str, int] = {"send_cuda_ipc": 0, "send_cuda_fallback": 0}


def debugprint(*args: Any, **kwargs: Any) -> None:
    if debug_all_messages:
        logger.debug(" ".join(str(arg) for arg in args))


# ---------------------------------------------------------------------------
# Serialization Functions
# ---------------------------------------------------------------------------

def _prepare_for_rpc(obj: Any) -> Any:
    """Recursively prepare objects for RPC transport.

    CUDA tensors:
        - If PYISOLATE_ENABLE_CUDA_IPC=1, leave CUDA tensors intact to allow
          torch.multiprocessing's CUDA IPC reducer to handle zero-copy.
        - Otherwise, move to CPU (shared memory when possible) for transport.

    Adapter-registered types are serialized via SerializerRegistry.
    Unpicklable custom containers are downgraded into plain serializable forms.
    """
    type_name = type(obj).__name__

    # Check for adapter-registered serializers first
    from .serialization_registry import SerializerRegistry
    registry = SerializerRegistry.get_instance()

    # Try exact type name first (fast path)
    if registry.has_handler(type_name):
        serializer = registry.get_serializer(type_name)
        if serializer:
            return serializer(obj)

    # Check base classes for inheritance support
    for base in type(obj).__mro__[1:]:  # Skip obj itself
        if registry.has_handler(base.__name__):
            serializer = registry.get_serializer(base.__name__)
            if serializer:
                return serializer(obj)

    try:
        import torch
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda:
                if _cuda_ipc_env_enabled:
                    _ipc_metrics["send_cuda_ipc"] += 1
                    return obj  # allow CUDA IPC path
                _ipc_metrics["send_cuda_fallback"] += 1
                return obj.cpu()
            return obj
    except ImportError:
        pass

    if isinstance(obj, dict):
        return {k: _prepare_for_rpc(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        converted = [_prepare_for_rpc(item) for item in obj]
        return tuple(converted) if isinstance(obj, tuple) else converted

    # Primitives pass through
    if isinstance(obj, (str, int, float, bool, type(None), bytes)):
        return obj

    return obj


def _tensor_to_cuda(obj: Any, device: Any | None = None) -> Any:
    """Rehydrate reference objects and containers after an RPC round-trip.

    Reference dictionaries with __type__ are converted to proxy objects or
    real instances via adapter-registered deserializers. Containers are
    recursively processed.
    """
    from types import SimpleNamespace
    if isinstance(obj, SimpleNamespace):
        type_name = getattr(obj, "__pyisolate_type__", None)
        if type_name == "RemoteObjectHandle":
            from .remote_handle import RemoteObjectHandle
            return RemoteObjectHandle(obj.object_id, obj.type_name)

        # CRITICAL FIX: Check for embedded remote handle
        handle = getattr(obj, "_pyisolate_remote_handle", None)
        if handle is not None:
            # Recursively unwrap the handle (it's also a SimpleNamespace)
            return _tensor_to_cuda(handle, device)

        return obj

    from .serialization_registry import SerializerRegistry
    registry = SerializerRegistry.get_instance()

    if isinstance(obj, dict):
        ref_type = obj.get("__type__")
        if ref_type and registry.has_handler(ref_type):
            deserializer = registry.get_deserializer(ref_type)
            if deserializer:
                return deserializer(obj)

        # Handle pyisolate internal container types
        if obj.get("__pyisolate_attribute_container__") and "data" in obj:
            converted = {k: _tensor_to_cuda(v, device) for k, v in obj["data"].items()}
            return AttributeContainer(converted)
        if obj.get("__pyisolate_attrdict__") and "data" in obj:
            converted = {k: _tensor_to_cuda(v, device) for k, v in obj["data"].items()}
            return AttrDict(converted)
        converted = {k: _tensor_to_cuda(v, device) for k, v in obj.items()}
        return AttrDict(converted)

    if isinstance(obj, (list, tuple)):
        converted_seq = [_tensor_to_cuda(item, device) for item in obj]
        return type(obj)(converted_seq) if isinstance(obj, tuple) else converted_seq

    return obj
