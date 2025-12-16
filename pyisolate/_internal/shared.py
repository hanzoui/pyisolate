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
- Handles custom ComfyUI types (CLIP, VAE, ModelPatcher) via proxies/registries
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
import contextvars
import inspect
import logging
import os
import queue
import threading
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ContextManager,
    Iterable,
    Literal,
    TypedDict,
    TypeVar,
    Union,
    cast,
    get_type_hints,
)

T = TypeVar("T")

from .model_serialization import serialize_for_isolation
if TYPE_CHECKING:
    ModelPatcherRegistry = Any
    ModelSamplingRegistry = Any
    ModelSamplingProxy = Any


class AttrDict(dict[str, Any]):
    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def copy(self) -> "AttrDict":
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

    def copy(self) -> "AttributeContainer":
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
else:
    import multiprocessing
    typehint_mp = multiprocessing

logger = logging.getLogger(__name__)

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

    Also converts ModelPatcher/ModelSampling objects into reference dictionaries
    and downgrades unpicklable custom containers into plain serializable forms.
    """
    type_name = type(obj).__name__

    # Custom serialization hook for ModelPatcher
    if type_name == 'ModelPatcher':
        import os
        if os.environ.get("PYISOLATE_CHILD") == "1":
            if not hasattr(obj, "_instance_id"):
                raise RuntimeError(
                    f"ModelPatcher {id(obj)} encountered in child process without _instance_id. "
                    f"This indicates the object was not properly proxied or the isolation is broken. "
                    f"All ModelPatcher objects in child processes must be proxies with _instance_id."
                )
            return {"__type__": "ModelPatcherRef", "model_id": getattr(obj, "_instance_id")}
        
        from comfy.isolation.model_patcher_proxy import ModelPatcherRegistry  # type: ignore[import-not-found]
        model_id = ModelPatcherRegistry().register(obj)
        return {"__type__": "ModelPatcherRef", "model_id": model_id}

    if type_name.startswith('ModelSampling'):
        import os
        if os.environ.get("PYISOLATE_CHILD") == "1":
            if not hasattr(obj, "_instance_id"):
                raise RuntimeError(
                    f"ModelSampling {id(obj)} encountered in child process without _instance_id. "
                    f"This indicates the object was not properly proxied or the isolation is broken. "
                    f"All ModelSampling objects in child processes must be proxies with _instance_id."
                )
            return {"__type__": "ModelSamplingRef", "ms_id": getattr(obj, "_instance_id")}
        
        from comfy.isolation.model_sampling_proxy import ModelSamplingRegistry  # type: ignore[import-not-found]
        ms_id = ModelSamplingRegistry().register(obj)
        return {"__type__": "ModelSamplingRef", "ms_id": ms_id}

    # Handle ModelPatcherProxy - convert to ref (child returning proxy to host)
    if type_name == 'ModelPatcherProxy':
        if not hasattr(obj, "_instance_id"):
            raise RuntimeError(
                f"ModelPatcherProxy {id(obj)} missing _instance_id. "
                f"This should never happen - all proxies are created with instance IDs."
            )
        return {"__type__": "ModelPatcherRef", "model_id": obj._instance_id}

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
    """Rehydrate ModelPatcher references and containers after an RPC round-trip.

    Previously this converted CPU tensors back to CUDA, but image tensors are
    expected to remain on CPU. Now this is a no-op passthrough except for reference
    objects and container recursion.
    """
    if isinstance(obj, dict) and obj.get("__type__") == "ModelPatcherRef":
        try:
            is_child = os.environ.get("PYISOLATE_CHILD") == "1"
            if is_child:
                from comfy.isolation.model_patcher_proxy import ModelPatcherProxy
                return ModelPatcherProxy(obj["model_id"], registry=None, manage_lifecycle=False)
            else:
                from comfy.isolation.model_patcher_proxy import ModelPatcherRegistry
                return ModelPatcherRegistry()._get_instance(obj["model_id"])
        except ImportError:
            pass

    if isinstance(obj, dict) and obj.get("__type__") in ("ModelPatcherOpaque", "ModelSamplingOpaque"):
        return obj

    if isinstance(obj, dict) and obj.get("__type__") == "ModelSamplingRef":
        try:
            is_child = os.environ.get("PYISOLATE_CHILD") == "1"
            if is_child:
                from comfy.isolation.model_sampling_proxy import ModelSamplingProxy
                return ModelSamplingProxy(obj["ms_id"])
            else:
                from comfy.isolation.model_sampling_proxy import ModelSamplingRegistry
                return ModelSamplingRegistry()._get_instance(obj["ms_id"])
        except ImportError:
            pass

    if isinstance(obj, dict):
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
    def __init__(
        self,
        recv_queue: typehint_mp.Queue[RPCMessage],
        send_queue: typehint_mp.Queue[RPCMessage],
    ):
        self.id = str(uuid.uuid4())
        self.handling_call_id: contextvars.ContextVar[int | None] = contextvars.ContextVar(
            self.id + "_handling_call_id", default=None)
        self.recv_queue = recv_queue
        self.send_queue = send_queue
        self.lock = threading.Lock()
        self.pending: dict[int, RPCPendingRequest] = {}
        self.default_loop = asyncio.get_event_loop()
        self.callees: dict[str, object] = {}
        self.callbacks: dict[str, Any] = {}
        self.blocking_future: asyncio.Future[Any] | None = None
        self.outbox: queue.Queue[RPCPendingRequest | None] = queue.Queue()

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
            # Explicit error handling for RPC dispatch failures.
            # Log full exception context (object/callback ID, stack trace) for debugging.
            # Convert exception to string for serialization across process boundary.
            logger.exception("RPC dispatch failed for %s", request.get("object_id", request.get("callback_id")))
            response = RPCResponse(kind="response", call_id=request["call_id"], result=None, error=str(exc))

        self.send_queue.put(_prepare_for_rpc(response))

    def _recv_thread(self) -> None:
        while True:
            try:
                item = _tensor_to_cuda(self.recv_queue.get())
            except Exception as exc:
                raise RuntimeError(f"RPC recv failed (rpc_id={self.id}): {exc}") from exc

            if item is None:
                if self.blocking_future:
                    self.default_loop.call_soon_threadsafe(self.blocking_future.set_result, None)
                break

            if item["kind"] == "response":
                with self.lock:
                    pending_request = self.pending.pop(item["call_id"], None)
                if pending_request:
                    if item.get("error"):
                        pending_request["calling_loop"].call_soon_threadsafe(
                            pending_request["future"].set_exception, Exception(item["error"]))
                    else:
                        pending_request["calling_loop"].call_soon_threadsafe(
                            pending_request["future"].set_result, item["result"])

            elif item["kind"] in ("call", "callback"):
                request = cast(Union[RPCRequest, RPCCallback], item)
                request_parent = request.get("parent_call_id")
                call_on_loop = self.default_loop

                if request_parent is not None:
                    with self.lock:
                        pending_request = self.pending.get(request_parent)
                    if pending_request:
                        call_on_loop = pending_request["calling_loop"]

                async def call_with_context(captured_request: RPCRequest | RPCCallback) -> None:
                    token = self.handling_call_id.set(captured_request["call_id"])
                    try:
                        return await self.dispatch_request(captured_request)
                    finally:
                        self.handling_call_id.reset(token)

                asyncio.run_coroutine_threadsafe(call_with_context(request), call_on_loop)

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
                    self.send_queue.put(request_msg)
                except Exception as exc:
                    with self.lock:
                        pending = self.pending.pop(call_id, None)
                    if pending:
                        pending["calling_loop"].call_soon_threadsafe(
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
                    self.send_queue.put(request_msg)
                except Exception as exc:
                    with self.lock:
                        pending = self.pending.pop(call_id, None)
                    if pending:
                        pending["calling_loop"].call_soon_threadsafe(
                            pending["future"].set_exception, RuntimeError(str(exc)))
                    raise

            elif typed_item["kind"] == "response":
                response_msg: RPCMessage = _prepare_for_rpc(typed_item)
                self.send_queue.put(response_msg)


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
