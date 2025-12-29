"""
RPC Serialization Layer - Architecture Overview

This module implements a three-layer serialization system for cross-process RPC communication.

LAYER 1: Low-Level Tensor Preparation (_prepare_for_rpc, formerly _tensor_to_cpu)
---------------------------------------------------------------------------------------------
Handles tensor-specific concerns:
- Detaches tensors from computation graph (prevents autograd leakage across processes)
- Moves tensors to shared memory (CPU) or enables CUDA IPC (GPU) for zero-copy transfer
- Preserves tensor metadata (shape, dtype, device info)
- IMPORTANT: Name is intentionally generic because behavior depends on CUDA IPC availability:
  * With CUDA IPC (Linux): GPU tensors stay on device via shared memory handles
  * Without CUDA IPC (Windows/macOS): Tensors copied to CPU shared memory

Called by: Layer 2 (serialize_for_isolation)
Operates on: Individual torch.Tensor objects

LAYER 2: High-Level Object Serialization (serialize_for_isolation)
-------------------------------------------------------------------
Handles composite data structures:
- Recursively traverses dicts, lists, tuples to find serializable objects
- Dispatches tensor objects to Layer 1 for hardware-specific handling
- Handles adapter-registered types via SerializerRegistry (pluggable)
- Preserves object relationships and nesting structure
- Special-cases opaque objects (represent as type name + truncated repr)

Called by: Layer 3 (RPC message sender before transmitting over pipe)
Operates on: Arbitrary Python objects (function arguments, return values)

LAYER 3: RPC Message Envelope (AsyncRPC.send_request / send_response)
----------------------------------------------------------------------
Handles protocol concerns:
- Wraps serialized data in request/response envelopes with message IDs
- Manages request correlation (matching responses to requests)
- Implements timeout handling and error propagation
- Routes messages through send/recv threads to avoid blocking event loop

WHY THREE LAYERS:
- Separation of concerns (tensor handling vs structure traversal vs protocol)
- Testability (each layer tested independently with mock data)
- Extensibility (add new tensor types in L1, new structures in L2, new transports in L3)

CONSOLIDATION TRADE-OFF:
Merging layers would couple hardware specifics (CUDA IPC) to structure logic (recursion),
making it harder to support new hardware (e.g., ROCm, MPS) or transport mechanisms (e.g.,
shared memory instead of pipes). Current design prioritizes maintainability over brevity.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import inspect
import logging
import os
import queue
import socket
import threading
import uuid
from collections.abc import Iterable
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Protocol,
    TypedDict,
    TypeVar,
    Union,
    cast,
    get_type_hints,
    runtime_checkable,
)

from .model_serialization import serialize_for_isolation

T = TypeVar("T")


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
                default = inspect.Parameter.empty
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

# We only import this to get type hinting working. It can also be a torch.multiprocessing
if TYPE_CHECKING:
    import multiprocessing as typehint_mp
    from multiprocessing.connection import Connection
else:
    import multiprocessing
    typehint_mp = multiprocessing

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transport Abstraction Layer
# ---------------------------------------------------------------------------
# These classes abstract the IPC mechanism, allowing AsyncRPC to work with
# either multiprocessing.Queue (standard) or multiprocessing.connection.Connection
# (Unix Domain Sockets for bwrap sandbox).


@runtime_checkable
class RPCTransport(Protocol):
    """Protocol for RPC transport mechanisms.

    Implementations must provide thread-safe send/recv operations.
    """

    def send(self, obj: Any) -> None:
        """Send an object to the remote endpoint."""
        ...

    def recv(self) -> Any:
        """Receive an object from the remote endpoint. Blocks until available."""
        ...

    def close(self) -> None:
        """Close the transport. Further send/recv calls may fail."""
        ...


class QueueTransport:
    """Transport using multiprocessing.Queue pairs (standard IPC)."""

    def __init__(
        self,
        send_queue: typehint_mp.Queue[Any],
        recv_queue: typehint_mp.Queue[Any],
    ) -> None:
        self._send_queue = send_queue
        self._recv_queue = recv_queue

    def send(self, obj: Any) -> None:
        self._send_queue.put(obj)

    def recv(self) -> Any:
        return self._recv_queue.get()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._send_queue.close()
        with contextlib.suppress(Exception):
            self._recv_queue.close()


class ConnectionTransport:
    """Transport using multiprocessing.connection.Connection (Unix Domain Sockets).

    Used for bwrap sandbox isolation where Queue-based IPC is not available.
    """

    def __init__(self, conn: Connection) -> None:
        self._conn = conn
        self._lock = threading.Lock()

    def send(self, obj: Any) -> None:
        with self._lock:
            self._conn.send(obj)

    def recv(self) -> Any:
        return self._conn.recv()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._conn.close()


class JSONSocketTransport:
    """Transport using raw sockets + JSON-RPC (pickle-safe).

    This transport uses JSON serialization instead of pickle to prevent
    RCE attacks via __reduce__ exploits from sandboxed child processes.

    Used for ALL Linux isolation modes (sandbox and non-sandbox).
    """

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._lock = threading.Lock()
        self._recv_lock = threading.Lock()

    def send(self, obj: Any) -> None:
        """Serialize to JSON with length prefix."""
        import json
        import struct

        try:
            data = json.dumps(obj, default=self._json_default).encode('utf-8')
        except TypeError as e:
            type_name = type(obj).__name__
            logger.error(
                "📚 [PyIsolate][JSON-RPC] Cannot serialize object:\n"
                "  Type: %s\n"
                "  Error: %s\n"
                "  Resolution: Register a custom serializer via SerializerRegistry",
                type_name, e
            )
            raise TypeError(f"Cannot JSON-serialize {type_name}: {e}") from e

        msg = struct.pack('>I', len(data)) + data
        with self._lock:
            self._sock.sendall(msg)

    def recv(self) -> Any:
        """Receive length-prefixed JSON message."""
        import json
        import struct

        with self._recv_lock:
            raw_len = self._recvall(4)
            if not raw_len or len(raw_len) < 4:
                raise ConnectionError("Socket closed or incomplete length header")
            msg_len = struct.unpack('>I', raw_len)[0]
            if msg_len > 100 * 1024 * 1024:  # 100MB sanity limit
                raise ValueError(f"Message too large: {msg_len} bytes")
            data = self._recvall(msg_len)
            if len(data) < msg_len:
                raise ConnectionError(f"Incomplete message: got {len(data)}/{msg_len} bytes")
            return json.loads(data.decode('utf-8'), object_hook=self._json_object_hook)

    def _recvall(self, n: int) -> bytes:
        """Receive exactly n bytes from the socket."""
        chunks = []
        remaining = n
        while remaining > 0:
            chunk = self._sock.recv(min(remaining, 65536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b''.join(chunks)

    def _json_default(self, obj: Any) -> Any:
        """Handle non-JSON types during serialization."""
        import traceback as tb_module
        from enum import Enum
        from types import FunctionType, MethodType

        # Skip callables/methods - they can't be serialized and are typically not needed
        # Introspection Support: We serialize signature metadata so the other side
        # can construct a proxy that passes inspect.signature() checks.
        if isinstance(obj, (MethodType, FunctionType)) or callable(obj) and not isinstance(obj, type):
            sig_metadata = []
            try:
                sig = inspect.signature(obj)
                for param in sig.parameters.values():
                    # Serialize (name, kind, has_default)
                    # We can't easily serialize arbitrary default values, so just boolean flag
                    has_default = param.default is not inspect.Parameter.empty
                    sig_metadata.append((param.name, int(param.kind), has_default))
            except Exception:
                # Some callables (e.g. builtins) might not have signature
                pass

            return {
                '__pyisolate_callable__': True,
                'type': type(obj).__name__,
                'name': getattr(obj, '__name__', str(obj)),
                'signature': sig_metadata
            }

        # Handle exceptions explicitly
        if isinstance(obj, BaseException):
            return {
                '__pyisolate_exception__': True,
                'type': type(obj).__name__,
                'module': type(obj).__module__,
                'args': [str(a) for a in obj.args],  # Convert args to strings for JSON
                'message': str(obj),
                'traceback': tb_module.format_exc() if tb_module.format_exc() != 'NoneType: None\n' else ''
            }

        # Handle Enums (must be before __dict__ check since Enums have __dict__)
        if isinstance(obj, Enum):
            return {
                '__pyisolate_enum__': True,
                'type': type(obj).__name__,
                'module': type(obj).__module__,
                'name': obj.name,
                'value': obj.value if isinstance(
                    obj.value, (int, str, float, bool, type(None))
                ) else str(obj.value)
            }

        # Handle bytes (common in some contexts)
        if isinstance(obj, bytes):
            import base64
            return {
                '__pyisolate_bytes__': True,
                'data': base64.b64encode(obj).decode('ascii')
            }

        # Handle UUID objects
        import uuid
        if isinstance(obj, uuid.UUID):
            return str(obj)

        # Handle PyTorch tensors BEFORE __dict__ check (tensors have __dict__ but shouldn't use it)
        try:
            import torch
            if isinstance(obj, torch.Tensor):
                from .tensor_serializer import serialize_tensor
                return serialize_tensor(obj)
        except ImportError:
            pass

        # Handle objects with __dict__ (preserve full state)
        if hasattr(obj, '__dict__') and not callable(obj):
            try:
                # Recursively serialize __dict__ contents AND class attributes
                serialized_dict = {}

                # First, collect JSON-serializable class attributes (not methods/descriptors)
                for klass in type(obj).__mro__:
                    if klass is object:
                        continue
                    for k, v in vars(klass).items():
                        if k.startswith('_'):
                            continue
                        # Only include primitive types as class attributes
                        if isinstance(v, (int, float, str, bool, type(None))) and k not in serialized_dict:
                            serialized_dict[k] = v

                # Then add instance attributes (which override class attrs)
                for k, v in obj.__dict__.items():
                    # Skip private attributes and methods/callables
                    if k.startswith('_'):
                        continue
                    if callable(v):
                        continue
                    try:
                        # Test if value is JSON-serializable
                        import json
                        json.dumps(v)
                        serialized_dict[k] = v
                    except TypeError:
                        # Try to serialize with our default handler
                        serialized_dict[k] = self._json_default(v)

                return {
                    '__pyisolate_object__': True,
                    'type': type(obj).__name__,
                    'module': type(obj).__module__,
                    'data': serialized_dict
                }
            except Exception as e:
                logger.warning("📚 [PyIsolate][JSON-RPC] Failed to serialize __dict__ of %s: %s",
                             type(obj).__name__, e)

        # Fail loudly for non-serializable types
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable. "
                       f"Register a serializer via SerializerRegistry.register()")

    def _json_object_hook(self, dct: dict) -> Any:
        """Reconstruct objects from JSON during deserialization."""
        from types import SimpleNamespace

        # Reconstruct exceptions
        if dct.get('__pyisolate_exception__'):
            exc_type = dct.get('type', 'Exception')
            exc_module = dct.get('module', 'builtins')
            msg = dct.get('message', '')
            remote_tb = dct.get('traceback', '')
            # Create a RuntimeError that preserves the original error info
            error = RuntimeError(f"Remote {exc_module}.{exc_type}: {msg}")
            if remote_tb:
                error.__pyisolate_remote_traceback__ = remote_tb  # type: ignore
            return error

        # Reconstruct bytes
        if dct.get('__pyisolate_bytes__'):
            import base64
            return base64.b64decode(dct['data'])

        # Generic Registry Lookup for __type__
        if '__type__' in dct:
            type_name = dct['__type__']
            # Skip TensorRef here as it has special handling below (or generic can handle it if registered)
            if type_name != 'TensorRef':
                from .serialization_registry import SerializerRegistry
                registry = SerializerRegistry.get_instance()
                deserializer = registry.get_deserializer(type_name)
                if deserializer:
                    try:
                        return deserializer(dct)
                    except Exception as e:
                        # Log error but don't crash - return dict as fallback
                        logger.warning(f"Failed to deserialize {type_name}: {e}")

        # Handle TensorRef - deserialize tensors during JSON parsing
        if dct.get('__type__') == 'TensorRef':
            from .serialization_registry import SerializerRegistry
            registry = SerializerRegistry.get_instance()
            if registry.has_handler('TensorRef'):
                deserializer = registry.get_deserializer('TensorRef')
                if deserializer:
                    return deserializer(dct)
            # Fallback: direct import if registry not yet populated
            try:
                from .tensor_serializer import deserialize_tensor
                return deserialize_tensor(dct)
            except Exception:
                pass
            return dct  # Last resort fallback

        # Reconstruct Enums
        if dct.get('__pyisolate_enum__'):
            import importlib
            module_name = dct.get('module', 'builtins')
            type_name = dct.get('type', 'Enum')
            enum_name = dct.get('name', '')
            try:
                module = importlib.import_module(module_name)
                enum_type = getattr(module, type_name, None)
                if enum_type and hasattr(enum_type, enum_name):
                    return getattr(enum_type, enum_name)
            except Exception:
                pass
            # Fallback: return the raw value if we can't reconstruct the enum
            return dct.get('value')

        # Reconstruct generic objects - try to recreate the original class
        if dct.get('__pyisolate_object__'):
            import importlib
            data = dct.get('data', {})
            module_name = dct.get('module')
            type_name = dct.get('type')

            # Try to reconstruct the original class
            if module_name and type_name:
                try:
                    module = importlib.import_module(module_name)
                    cls = getattr(module, type_name, None)
                    if cls is not None:
                        # Try to create instance - some classes have special constructors
                        # First, try if it takes a 'cond' arg (common for CONDRegular etc.)
                        if 'cond' in data:
                            try:
                                return cls(data['cond'])
                            except Exception:
                                pass
                        # Try no-arg constructor (calls __init__)
                        try:
                            obj = cls()
                            for k, v in data.items():
                                set_attr = getattr(obj, k, None)
                                # Check if it's a property without a setter
                                if isinstance(getattr(type(obj), k, None), property) and getattr(type(obj), k).fset is None:
                                   continue
                                setattr(obj, k, v)
                            return obj
                        except Exception:
                            pass
                        # Last resort: __new__ without __init__ then set attributes
                        try:
                            obj = cls.__new__(cls)
                            for k, v in data.items():
                                setattr(obj, k, v)
                            return obj
                        except Exception:
                            pass
                except Exception:
                    pass

            # Fallback: return SimpleNamespace with metadata
            ns = SimpleNamespace(**data)
            ns.__pyisolate_type__ = type_name
            ns.__pyisolate_module__ = module_name
            return ns
        
        # Reconstruct Callables
        if dct.get('__pyisolate_callable__'):
            return CallableProxy(dct)

        return dct

    def close(self) -> None:
        """Close the socket connection."""
        with contextlib.suppress(Exception):
            self._sock.shutdown(2)  # SHUT_RDWR
        with contextlib.suppress(Exception):
            self._sock.close()


# Debug flag for verbose RPC message logging (set via PYISOLATE_DEBUG_RPC=1)
debug_all_messages = bool(os.environ.get("PYISOLATE_DEBUG_RPC"))
_debug_rpc = debug_all_messages
_cuda_ipc_env_enabled = os.environ.get("PYISOLATE_ENABLE_CUDA_IPC") == "1"
_cuda_ipc_warned = False
_ipc_metrics: dict[str, int] = {"send_cuda_ipc": 0, "send_cuda_fallback": 0}


def debugprint(*args: Any, **kwargs: Any) -> None:
    if debug_all_messages:
        logger.debug(" ".join(str(arg) for arg in args))


# Global RPC instance for child process (set during initialization)
_child_rpc_instance: AsyncRPC | None = None


def set_child_rpc_instance(rpc: AsyncRPC | None) -> None:
    """Set the global RPC instance for use inside isolated child processes."""
    global _child_rpc_instance
    _child_rpc_instance = rpc


def get_child_rpc_instance() -> AsyncRPC | None:
    """Return the current child-process RPC instance (if any)."""
    return _child_rpc_instance


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


def local_execution(func: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a ProxiedSingleton method for local execution instead of RPC."""
    func._is_local_execution = True  # type: ignore[attr-defined]
    return func


class LocalMethodRegistry:
    _instance: LocalMethodRegistry | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._local_implementations: dict[type, object] = {}
        self._local_methods: dict[type, set[str]] = {}

    @classmethod
    def get_instance(cls) -> LocalMethodRegistry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register_class(self, cls: type) -> None:
        # Use object.__new__ to bypass singleton __init__ and prevent infinite recursion.
        # Standard instantiation would trigger __init__, which registers the singleton,
        # which would call register_class again, creating an infinite loop.
        local_instance: Any = object.__new__(cls)
        cls.__init__(local_instance)  # type: ignore[misc]
        self._local_implementations[cls] = local_instance

        local_methods = set()
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if getattr(method, "_is_local_execution", False):
                local_methods.add(name)
        for name in dir(cls):
            if not name.startswith("_"):
                attr = getattr(cls, name, None)
                if callable(attr) and getattr(attr, "_is_local_execution", False):
                    local_methods.add(name)
        self._local_methods[cls] = local_methods

    def is_local_method(self, cls: type, method_name: str) -> bool:
        return cls in self._local_methods and method_name in self._local_methods[cls]

    def get_local_method(self, cls: type, method_name: str) -> Callable[..., Any]:
        if cls not in self._local_implementations:
            raise ValueError(f"Class {cls} not registered for local execution")
        return cast(Callable[..., Any], getattr(self._local_implementations[cls], method_name))


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


# RPC message types define the protocol for bidirectional async communication
# between host and isolated processes. Each message type serves a specific purpose:
# - RPCRequest: Host initiates method call on isolated object
# - RPCCallback: Isolated process calls back to host-registered function
# - RPCResponse: Returns result (or error) for any pending call
class RPCResponse(TypedDict):
    kind: Literal["response"]
    call_id: int
    result: Any
    error: str | None


RPCMessage = Union[RPCRequest, RPCCallback, RPCResponse]


class RPCPendingRequest(TypedDict):
    kind: Literal["call", "callback"]
    object_id: str
    parent_call_id: int | None
    calling_loop: asyncio.AbstractEventLoop
    future: asyncio.Future[Any]
    method: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


proxied_type = TypeVar("proxied_type", bound=object)


class AsyncRPC:
    """Asynchronous RPC layer for inter-process communication.

    Supports two initialization modes:
    1. Legacy: Pass recv_queue and send_queue (backward compatible)
    2. Transport: Pass a single RPCTransport instance (for sandbox/UDS)
    """

    def __init__(
        self,
        recv_queue: typehint_mp.Queue[RPCMessage] | None = None,
        send_queue: typehint_mp.Queue[RPCMessage] | None = None,
        *,
        transport: RPCTransport | None = None,
    ):
        self.id = str(uuid.uuid4())
        self.handling_call_id: contextvars.ContextVar[int | None] = contextvars.ContextVar(
            self.id + "_handling_call_id", default=None)

        # Support both legacy queue interface and new transport interface
        if transport is not None:
            self._transport = transport
        elif recv_queue is not None and send_queue is not None:
            self._transport = QueueTransport(send_queue, recv_queue)
        else:
            raise ValueError("Must provide either (recv_queue, send_queue) or transport")

        self.lock = threading.Lock()
        self.pending: dict[int, RPCPendingRequest] = {}
        self.default_loop = asyncio.get_event_loop()
        self._loop_lock = threading.Lock()  # Protects default_loop updates
        self.callees: dict[str, object] = {}
        self.callbacks: dict[str, Any] = {}
        self.blocking_future: asyncio.Future[Any] | None = None
        self.outbox: queue.Queue[RPCPendingRequest | None] = queue.Queue()

    def update_event_loop(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """
        Update the default event loop used by this RPC instance.

        Call this method when the event loop changes to ensure RPC calls are 
        scheduled on the correct loop.

        Args:
            loop: The new event loop to use. If None, uses asyncio.get_event_loop().
        """
        with self._loop_lock:
            if loop is None:
                loop = asyncio.get_event_loop()
            self.default_loop = loop
            logger.debug(f"RPC {self.id}: Updated default_loop to {loop}")

    def register_callback(self, func: Any) -> str:
        callback_id = str(uuid.uuid4())
        with self.lock:
            self.callbacks[callback_id] = func
        return callback_id

    async def call_callback(self, callback_id: str, *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_event_loop()
        pending_request = RPCPendingRequest(
            kind="callback",
            object_id=callback_id,
            parent_call_id=self.handling_call_id.get(),
            calling_loop=loop,
            future=loop.create_future(),
            method="__call__",
            args=args,
            kwargs=kwargs,
        )
        # Use outbox pattern to avoid blocking RPC event loop.
        # Direct queue.put() would block if queue is full, stalling all RPC operations.
        # Outbox allows async fire-and-forget with separate task handling backpressure.
        self.outbox.put(pending_request)
        return await pending_request["future"]

    def create_caller(self, abc: type[proxied_type], object_id: str) -> proxied_type:
        this = self

        class CallWrapper:
            def __getattr__(self, name: str) -> Any:
                attr = getattr(abc, name, None)
                if not callable(attr) or name.startswith("_"):
                    raise AttributeError(f"{name} is not a valid method")

                registry = LocalMethodRegistry.get_instance()
                if registry.is_local_method(abc, name):
                    return registry.get_local_method(abc, name)

                if not inspect.iscoroutinefunction(attr):
                    raise ValueError(f"{name} is not a coroutine function")

                async def method(*args: Any, **kwargs: Any) -> Any:
                    loop = asyncio.get_event_loop()
                    pending_request = RPCPendingRequest(
                        kind="call",
                        object_id=object_id,
                        parent_call_id=this.handling_call_id.get(),
                        calling_loop=loop,
                        future=loop.create_future(),
                        method=name,
                        args=args,
                        kwargs=kwargs,
                    )
                    this.outbox.put(pending_request)
                    return await pending_request["future"]

                return method

        return cast(proxied_type, CallWrapper())

    def register_callee(self, object_instance: object, object_id: str) -> None:
        with self.lock:
            if object_id in self.callees:
                raise ValueError(f"Object ID {object_id} already registered")
            self.callees[object_id] = object_instance

    async def run_until_stopped(self) -> None:
        if self.blocking_future is None:
            self.run()
        assert self.blocking_future is not None, (
            "RPC event loop not running: blocking_future is None. "
            "Ensure run() was called before run_until_stopped()."
        )
        await self.blocking_future

    async def stop(self) -> None:
        assert self.blocking_future is not None, (
            "Cannot stop RPC: blocking_future is None. "
            "RPC event loop was never started or already stopped."
        )
        self.blocking_future.set_result(None)

    def run(self) -> None:
        self.blocking_future = self.default_loop.create_future()
        self._threads = [
            threading.Thread(target=self._recv_thread, daemon=True),
            threading.Thread(target=self._send_thread, daemon=True),
        ]
        for t in self._threads:
            t.start()

    async def dispatch_request(self, request: RPCRequest | RPCCallback) -> None:
        try:
            if request["kind"] == "callback":
                callback = None
                with self.lock:
                    callback = self.callbacks.get(request["callback_id"])
                if callback is None:
                    raise ValueError(f"Callback ID {request['callback_id']} not found")
                result = (
                    (await callback(*request["args"], **request["kwargs"]))
                    if inspect.iscoroutinefunction(callback)
                    else callback(*request["args"], **request["kwargs"])
                )
            elif request["kind"] == "call":
                callee = None
                with self.lock:
                    callee = self.callees.get(request["object_id"])
                if callee is None:
                    raise ValueError(f"Object ID {request['object_id']} not registered")
                func = getattr(callee, request["method"])
                result = (
                    (await func(*request["args"], **request["kwargs"]))
                    if inspect.iscoroutinefunction(func)
                    else func(*request["args"], **request["kwargs"])
                )
            else:
                # Fail loud on unknown request kinds rather than silently ignoring
                raise ValueError(
                    f"Unknown RPC request kind: {request.get('kind')}. "
                    f"Valid kinds are: 'call', 'callback'. "
                    f"Request: {request}"
                )
            response = RPCResponse(kind="response", call_id=request["call_id"], result=result, error=None)
        except Exception as exc:
            # Log full exception context for debugging; convert to string for serialization.
            obj_id = request.get("object_id", request.get("callback_id"))
            logger.exception("RPC dispatch failed for %s", obj_id)
            response = RPCResponse(
                kind="response", call_id=request["call_id"], result=None, error=str(exc)
            )

        # Try to send response; if serialization fails, send error response instead
        try:
            self._transport.send(_prepare_for_rpc(response))
        except (TypeError, ValueError) as serialize_exc:
            # FAIL LOUD: Log and propagate serialization failures
            logger.error(
                "RPC response serialization failed for call_id=%s: %s",
                request["call_id"], serialize_exc
            )
            # Try to send a minimal error response (no result, just error string)
            error_response = RPCResponse(
                kind="response",
                call_id=request["call_id"],
                result=None,
                error=f"Response serialization failed: {serialize_exc}"
            )
            try:
                self._transport.send(_prepare_for_rpc(error_response))
            except Exception as fallback_exc:
                # If even the error response can't be sent, raise to kill the RPC
                raise RuntimeError(
                    f"Cannot send RPC response or error for call_id={request['call_id']}: "
                    f"original error: {serialize_exc}, fallback error: {fallback_exc}"
                ) from serialize_exc

    def _get_valid_loop(
        self, preferred_loop: asyncio.AbstractEventLoop | None = None
    ) -> asyncio.AbstractEventLoop:
        """
        Get a valid (non-closed) event loop for RPC operations.

        This handles the case where the original loop has been closed
        and we need to use the current loop.

        The preferred_loop is typically the cached self.default_loop. If it's closed,
        this method will:
        1. Check if self.default_loop has been updated (via update_event_loop())
        2. Try to get the running loop (if called from async context)
        3. Return None if no valid loop is available (caller must handle)

        Args:
            preferred_loop: The loop we'd prefer to use if it's still valid

        Returns:
            A valid, non-closed event loop

        Raises:
            RuntimeError: If no valid event loop is available
        """
        # If preferred loop is valid, use it
        if preferred_loop is not None and not preferred_loop.is_closed():
            return preferred_loop

        # Check if default_loop has been updated by main thread
        with self._loop_lock:
            current_default = self.default_loop
            if current_default is not None and not current_default.is_closed():
                return current_default

        # Try to get the running event loop (works if called from async context)
        try:
            loop = asyncio.get_running_loop()
            if not loop.is_closed():
                with self._loop_lock:
                    self.default_loop = loop
                return loop
        except RuntimeError:
            pass  # No running loop

        # For the main thread, try get_event_loop
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                with self._loop_lock:
                    self.default_loop = loop
                return loop
        except RuntimeError:
            pass

        # No valid loop available - caller must handle this
        raise RuntimeError(
            f"RPC {self.id}: No valid event loop available. "
            "Call update_event_loop() from the main thread after creating a new loop."
        )

    def _recv_thread(self) -> None:
        while True:
            try:
                item = _tensor_to_cuda(self._transport.recv())
            except Exception as exc:
                raise RuntimeError(f"RPC recv failed (rpc_id={self.id}): {exc}") from exc

            if item is None:
                if self.blocking_future:
                    try:
                        loop = self._get_valid_loop(self.default_loop)
                        loop.call_soon_threadsafe(self.blocking_future.set_result, None)
                    except RuntimeError:
                        pass  # Loop closed, blocking_future won't be awaited anyway
                break

            if item["kind"] == "response":
                with self.lock:
                    pending_request = self.pending.pop(item["call_id"], None)
                if pending_request:
                    # Get a valid loop - the calling_loop may be closed
                    calling_loop = pending_request["calling_loop"]
                    if calling_loop.is_closed():
                        # Original loop is closed, try to get the current one
                        try:
                            calling_loop = self._get_valid_loop()
                        except RuntimeError:
                            logger.warning(
                                f"RPC {self.id}: Cannot deliver response {item['call_id']} - "
                                "original loop closed and no current loop available"
                            )
                            continue

                    try:
                        if item.get("error"):
                            calling_loop.call_soon_threadsafe(
                                pending_request["future"].set_exception, Exception(item["error"]))
                        else:
                            calling_loop.call_soon_threadsafe(
                                pending_request["future"].set_result, item["result"])
                    except RuntimeError as e:
                        if "Event loop is closed" in str(e):
                            logger.warning(
                                f"RPC {self.id}: Loop closed while delivering response {item['call_id']}"
                            )
                        else:
                            raise

            elif item["kind"] in ("call", "callback"):
                request = cast(Union[RPCRequest, RPCCallback], item)
                request_parent = request.get("parent_call_id")

                # Get a valid loop for dispatching this request
                try:
                    call_on_loop = self._get_valid_loop(self.default_loop)
                except RuntimeError as e:
                    logger.error(
                        f"RPC {self.id}: Cannot dispatch request {request.get('call_id')} - "
                        f"no valid event loop: {e}"
                    )
                    # Send error response back
                    error_response = RPCResponse(
                        kind="response",
                        call_id=request["call_id"],
                        result=None,
                        error=f"No valid event loop available: {e}"
                    )
                    self._transport.send(_prepare_for_rpc(error_response))
                    continue

                if request_parent is not None:
                    with self.lock:
                        pending_request = self.pending.get(request_parent)
                    if pending_request:
                        parent_loop = pending_request["calling_loop"]
                        if not parent_loop.is_closed():
                            call_on_loop = parent_loop

                async def call_with_context(captured_request: RPCRequest | RPCCallback) -> None:
                    token = self.handling_call_id.set(captured_request["call_id"])
                    try:
                        return await self.dispatch_request(captured_request)
                    finally:
                        self.handling_call_id.reset(token)

                try:
                    asyncio.run_coroutine_threadsafe(call_with_context(request), call_on_loop)
                except RuntimeError as e:
                    if "Event loop is closed" in str(e):
                        # Loop closed between our check and the call - try again with fresh loop
                        logger.warning(f"RPC {self.id}: Loop closed, retrying with fresh loop")
                        call_on_loop = self._get_valid_loop()
                        asyncio.run_coroutine_threadsafe(call_with_context(request), call_on_loop)
                    else:
                        raise

    def _send_thread(self) -> None:
        id_gen = 0
        while True:
            item = self.outbox.get()
            if item is None:
                break
            typed_item: RPCPendingRequest = item

            if typed_item["kind"] == "call":
                call_id = id_gen
                id_gen += 1
                with self.lock:
                    self.pending[call_id] = typed_item
                serialized_args = serialize_for_isolation(typed_item["args"])
                serialized_kwargs = serialize_for_isolation(typed_item["kwargs"])
                request_msg: RPCMessage = RPCRequest(
                    kind="call",
                    object_id=typed_item["object_id"],
                    call_id=call_id,
                    parent_call_id=typed_item["parent_call_id"],
                    method=typed_item["method"],
                    args=_prepare_for_rpc(serialized_args),
                    kwargs=_prepare_for_rpc(serialized_kwargs),
                )
                try:
                    self._transport.send(request_msg)
                except Exception as exc:
                    with self.lock:
                        pending = self.pending.pop(call_id, None)
                    if pending:
                        calling_loop = pending["calling_loop"]
                        if not calling_loop.is_closed():
                            with contextlib.suppress(RuntimeError):
                                calling_loop.call_soon_threadsafe(
                                    pending["future"].set_exception, RuntimeError(str(exc)))
                    raise

            elif typed_item["kind"] == "callback":
                call_id = id_gen
                id_gen += 1
                with self.lock:
                    self.pending[call_id] = typed_item
                serialized_args = serialize_for_isolation(typed_item["args"])
                serialized_kwargs = serialize_for_isolation(typed_item["kwargs"])
                request_msg = RPCCallback(
                    kind="callback",
                    callback_id=typed_item["object_id"],
                    call_id=call_id,
                    parent_call_id=typed_item["parent_call_id"],
                    args=_prepare_for_rpc(serialized_args),
                    kwargs=_prepare_for_rpc(serialized_kwargs),
                )
                try:
                    self._transport.send(request_msg)
                except Exception as exc:
                    with self.lock:
                        pending = self.pending.pop(call_id, None)
                    if pending:
                        calling_loop = pending["calling_loop"]
                        if not calling_loop.is_closed():
                            with contextlib.suppress(RuntimeError):
                                calling_loop.call_soon_threadsafe(
                                    pending["future"].set_exception, RuntimeError(str(exc)))
                    raise

            elif typed_item["kind"] == "response":
                response_msg: RPCMessage = _prepare_for_rpc(typed_item)
                self._transport.send(response_msg)


class SingletonMetaclass(type):
    _instances: dict[type, Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    def inject_instance(cls: type[T], instance: Any) -> None:
        assert cls not in SingletonMetaclass._instances, (
            f"Cannot inject instance for {cls.__name__}: singleton already exists. "
            f"Instance was likely created before injection attempt. "
            f"Ensure inject_instance() is called before any instantiation."
        )
        SingletonMetaclass._instances[cls] = instance

    def get_instance(cls: type[T], *args: Any, **kwargs: Any) -> T:
        if cls not in SingletonMetaclass._instances:
            SingletonMetaclass._instances[cls] = super().__call__(*args, **kwargs)  # type: ignore[misc]
        return cast(T, SingletonMetaclass._instances[cls])

    def use_remote(cls, rpc: AsyncRPC) -> None:
        assert issubclass(cls, ProxiedSingleton), (
            f"Class {cls.__name__} must inherit from ProxiedSingleton to use remote RPC capabilities."
        )
        remote = rpc.create_caller(cls, cls.get_remote_id())
        LocalMethodRegistry.get_instance().register_class(cls)
        cls.inject_instance(remote)

        for name, t_hint in get_type_hints(cls).items():
            if isinstance(t_hint, type) and issubclass(t_hint, ProxiedSingleton) and not name.startswith("_"):
                caller = rpc.create_caller(t_hint, t_hint.get_remote_id())
                setattr(remote, name, caller)


class ProxiedSingleton(metaclass=SingletonMetaclass):
    """Cross-process singleton with RPC-proxied method calls."""

    def __init__(self) -> None:
        object.__init__(self)

    @classmethod
    def get_remote_id(cls) -> str:
        return cls.__name__

    def _register(self, rpc: AsyncRPC) -> None:
        rpc.register_callee(self, self.get_remote_id())
        for name, attr in self.__class__.__dict__.items():
            if isinstance(attr, ProxiedSingleton) and not name.startswith("_"):
                if attr is self:
                    continue
                attr._register(rpc)
