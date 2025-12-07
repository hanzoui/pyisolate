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
    Literal,
    TypedDict,
    TypeVar,
    Union,
    cast,
    get_type_hints,
)

if TYPE_CHECKING:
    import multiprocessing as typehint_mp
else:
    import multiprocessing
    typehint_mp = multiprocessing

logger = logging.getLogger(__name__)

_debug_rpc = bool(os.environ.get("PYISOLATE_DEBUG_RPC"))
_child_rpc_instance: Any = None


def set_child_rpc_instance(rpc: Any) -> None:
    global _child_rpc_instance
    _child_rpc_instance = rpc


def get_child_rpc_instance() -> Any:
    return _child_rpc_instance


def _tensor_to_cpu(obj: Any) -> Any:
    type_name = type(obj).__name__

    if type_name == 'ModelPatcher':
        try:
            from comfy.isolation.model_patcher_proxy import ModelPatcherRegistry
            model_id = ModelPatcherRegistry().register(obj)
            return {"__type__": "ModelPatcherRef", "model_id": model_id}
        except ImportError:
            pass

    if type_name == 'ModelPatcherProxy':
        try:
            return {"__type__": "ModelPatcherRef", "model_id": obj._instance_id}
        except Exception:
            pass

    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.cpu() if obj.is_cuda else obj
    except ImportError:
        pass

    if isinstance(obj, dict):
        dict_class = type(obj)
        dict_module = getattr(dict_class, '__module__', '') or ''
        if dict_class is not dict and dict_module and not dict_module.startswith(
                ('builtins', 'comfy', 'pyisolate', 'torch', 'numpy', 'typing')):
            return {k: _tensor_to_cpu(v) for k, v in obj.items()}
        return {k: _tensor_to_cpu(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        converted = [_tensor_to_cpu(item) for item in obj]
        return tuple(converted) if isinstance(obj, tuple) else converted

    if isinstance(obj, (str, int, float, bool, type(None), bytes)):
        return obj

    obj_module = getattr(type(obj), '__module__', '') or ''
    if obj_module and not obj_module.startswith(
            ('builtins', 'comfy', 'pyisolate', 'torch', 'numpy', 'typing', 'collections', 'abc')):
        return {
            "__pyisolate_unpicklable__": True,
            "type": type(obj).__name__,
            "module": obj_module,
            "repr": str(obj)[:1000],
        }

    return obj


def _tensor_to_cuda(obj: Any, device: Any = None) -> Any:
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

    if isinstance(obj, dict):
        return {k: _tensor_to_cuda(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_tensor_to_cuda(item, device) for item in obj]
        return type(obj)(converted) if isinstance(obj, tuple) else converted

    return obj


def local_execution(func):
    """Mark a ProxiedSingleton method for local execution instead of RPC."""
    func._is_local_execution = True
    return func


class LocalMethodRegistry:
    _instance: LocalMethodRegistry | None = None
    _lock = threading.Lock()

    def __init__(self):
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
        local_instance = object.__new__(cls)
        cls.__init__(local_instance)
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

    def get_local_method(self, cls: type, method_name: str):
        if cls not in self._local_implementations:
            raise ValueError(f"Class {cls} not registered for local execution")
        return getattr(self._local_implementations[cls], method_name)


class RPCRequest(TypedDict):
    kind: Literal["call"]
    object_id: str
    call_id: int
    parent_call_id: int | None
    method: str
    args: tuple
    kwargs: dict


class RPCCallback(TypedDict):
    kind: Literal["callback"]
    callback_id: str
    call_id: int
    parent_call_id: int | None
    args: tuple
    kwargs: dict


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
    future: asyncio.Future
    method: str
    args: tuple
    kwargs: dict


proxied_type = TypeVar("proxied_type", bound=object)
current_rpc_context: contextvars.ContextVar[AsyncRPC | None] = contextvars.ContextVar(
    "current_rpc_context", default=None)


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
        self.blocking_future: asyncio.Future | None = None
        self.outbox: queue.Queue[RPCPendingRequest] = queue.Queue()

    def register_callback(self, func: Any) -> str:
        callback_id = str(uuid.uuid4())
        with self.lock:
            self.callbacks[callback_id] = func
        return callback_id

    async def call_callback(self, callback_id: str, *args, **kwargs):
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
        self.outbox.put(pending_request)
        return await pending_request["future"]

    def create_caller(self, abc: type[proxied_type], object_id: str) -> proxied_type:
        this = self

        class CallWrapper:
            def __getattr__(self, name):
                attr = getattr(abc, name, None)
                if not callable(attr) or name.startswith("_"):
                    raise AttributeError(f"{name} is not a valid method")

                registry = LocalMethodRegistry.get_instance()
                if registry.is_local_method(abc, name):
                    return registry.get_local_method(abc, name)

                if not inspect.iscoroutinefunction(attr):
                    raise ValueError(f"{name} is not a coroutine function")

                async def method(*args, **kwargs):
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

    def register_callee(self, object_instance: object, object_id: str):
        with self.lock:
            if object_id in self.callees:
                raise ValueError(f"Object ID {object_id} already registered")
            self.callees[object_id] = object_instance

    async def run_until_stopped(self):
        if self.blocking_future is None:
            self.run()
        assert self.blocking_future is not None
        await self.blocking_future

    async def stop(self):
        assert self.blocking_future is not None
        self.blocking_future.set_result(None)

    def run(self):
        self.blocking_future = self.default_loop.create_future()
        self._threads = [
            threading.Thread(target=self._recv_thread, daemon=True),
            threading.Thread(target=self._send_thread, daemon=True),
        ]
        for t in self._threads:
            t.start()

    async def dispatch_request(self, request: RPCRequest | RPCCallback):
        token = current_rpc_context.set(self)
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
            else:
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
            response = RPCResponse(kind="response", call_id=request["call_id"], result=result, error=None)
        except Exception as exc:
            logger.exception("RPC dispatch failed for %s", request.get("object_id", request.get("callback_id")))
            response = RPCResponse(kind="response", call_id=request["call_id"], result=None, error=str(exc))
        finally:
            current_rpc_context.reset(token)

        self.send_queue.put(_tensor_to_cpu(response))

    def _recv_thread(self):
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

                async def call_with_context(captured_request: RPCRequest | RPCCallback):
                    token = self.handling_call_id.set(captured_request["call_id"])
                    try:
                        return await self.dispatch_request(captured_request)
                    finally:
                        self.handling_call_id.reset(token)

                asyncio.run_coroutine_threadsafe(call_with_context(request), call_on_loop)

    def _send_thread(self):
        id_gen = 0
        while True:
            item = self.outbox.get()
            if item is None:
                break

            if item["kind"] == "call":
                call_id = id_gen
                id_gen += 1
                with self.lock:
                    self.pending[call_id] = item
                request = RPCRequest(
                    kind="call",
                    object_id=item["object_id"],
                    call_id=call_id,
                    parent_call_id=item["parent_call_id"],
                    method=item["method"],
                    args=_tensor_to_cpu(item["args"]),
                    kwargs=_tensor_to_cpu(item["kwargs"]),
                )
                try:
                    self.send_queue.put(request)
                except Exception as exc:
                    with self.lock:
                        pending = self.pending.pop(call_id, None)
                    if pending:
                        pending["calling_loop"].call_soon_threadsafe(
                            pending["future"].set_exception, RuntimeError(str(exc)))
                    raise

            elif item["kind"] == "callback":
                call_id = id_gen
                id_gen += 1
                with self.lock:
                    self.pending[call_id] = item
                request = RPCCallback(
                    kind="callback",
                    callback_id=item["object_id"],
                    call_id=call_id,
                    parent_call_id=item["parent_call_id"],
                    args=_tensor_to_cpu(item["args"]),
                    kwargs=_tensor_to_cpu(item["kwargs"]),
                )
                try:
                    self.send_queue.put(request)
                except Exception as exc:
                    with self.lock:
                        pending = self.pending.pop(call_id, None)
                    if pending:
                        pending["calling_loop"].call_soon_threadsafe(
                            pending["future"].set_exception, RuntimeError(str(exc)))
                    raise

            elif item["kind"] == "response":
                self.send_queue.put(_tensor_to_cpu(item))


class SingletonMetaclass(type):
    T = TypeVar("T", bound="SingletonMetaclass")
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    def inject_instance(cls: type[T], instance: T) -> None:
        assert cls not in SingletonMetaclass._instances
        SingletonMetaclass._instances[cls] = instance

    def get_instance(cls: type[T], *args, **kwargs) -> T:
        if cls not in SingletonMetaclass._instances:
            SingletonMetaclass._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    def use_remote(cls, rpc: AsyncRPC) -> None:
        assert issubclass(cls, ProxiedSingleton)
        remote = rpc.create_caller(cls, cls.get_remote_id())
        LocalMethodRegistry.get_instance().register_class(cls)
        cls.inject_instance(remote)

        for name, t in get_type_hints(cls).items():
            if isinstance(t, type) and issubclass(t, ProxiedSingleton) and not name.startswith("_"):
                caller = rpc.create_caller(t, t.get_remote_id())
                setattr(remote, name, caller)


class ProxiedSingleton(metaclass=SingletonMetaclass):
    """Cross-process singleton with RPC-proxied method calls."""

    def __init__(self):
        super().__init__()

    @classmethod
    def get_remote_id(cls) -> str:
        return cls.__name__

    def _register(self, rpc: AsyncRPC):
        rpc.register_callee(self, self.get_remote_id())
        for name, attr in self.__class__.__dict__.items():
            if isinstance(attr, ProxiedSingleton) and not name.startswith("_"):
                if attr is self:
                    continue
                attr._register(rpc)
