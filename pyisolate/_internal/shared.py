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

# We only import this to get type hinting working. It can also be a torch.multiprocessing
if TYPE_CHECKING:
    import multiprocessing as typehint_mp
else:
    import multiprocessing

    typehint_mp = multiprocessing

logger = logging.getLogger(__name__)

# Debug flag for verbose RPC message logging (set via PYISOLATE_DEBUG_RPC=1)
debug_all_messages = bool(os.environ.get("PYISOLATE_DEBUG_RPC"))


def debugprint(*args, **kwargs):
    if debug_all_messages:
        logger.debug(" ".join(str(arg) for arg in args))


# Global RPC instance for child process (set during initialization)
_child_rpc_instance: Any = None  # Optional[AsyncRPC] but avoid circular import

def set_child_rpc_instance(rpc: Any) -> None:
    """Set the global RPC instance for child process (internal use only)."""
    global _child_rpc_instance
    _child_rpc_instance = rpc

def get_child_rpc_instance() -> Any:
    """Get the global RPC instance for child process (internal use only)."""
    return _child_rpc_instance


def _tensor_to_cpu(obj: Any) -> Any:
    """
    Recursively convert CUDA tensors to CPU tensors for safe IPC serialization.
    This avoids cudaMallocAsync IPC sharing issues.
    
    Also handles custom object serialization (e.g., ModelPatcher for ComfyUI).
    """
    type_name = type(obj).__name__
    
    # Custom serialization hook for ModelPatcher (ComfyUI integration)
    if type_name == 'ModelPatcher':
        try:
            from comfy.isolation.model_patcher_proxy import ModelPatcherRegistry
            registry = ModelPatcherRegistry()
            model_id = registry.register(obj)
            logger.info(f"📚 [PyIsolate][Serialization] ModelPatcher → ref {model_id}")
            return {
                "__type__": "ModelPatcherRef",
                "model_id": model_id,
            }
        except ImportError as e:
            logger.warning(f"📚 [PyIsolate][Serialization] ComfyUI integration not available: {e}")
        except Exception as e:
            logger.error(f"📚 [PyIsolate][Serialization] Failed to serialize ModelPatcher: {e}")
    
    # Handle ModelPatcherProxy - convert to ref (child returning proxy to host)
    if type_name == 'ModelPatcherProxy':
        try:
            # Proxy already has an instance_id - just convert to ref
            model_id = obj._instance_id
            logger.info(f"📚 [PyIsolate][Serialization] ModelPatcherProxy → ref {model_id}")
            return {
                "__type__": "ModelPatcherRef",
                "model_id": model_id,
            }
        except Exception as e:
            logger.error(f"📚 [PyIsolate][Serialization] Failed to serialize ModelPatcherProxy: {e}")
    
    # Also check for model_sampling attribute (nested pickle issue)
    if hasattr(obj, 'model_sampling') and type(obj.model_sampling).__name__ == 'ModelSampling':
        logger.warning(f"📚 [PyIsolate][Serialization] Detected ModelSampling in object, this may cause pickle issues")
    
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda:
                return obj.cpu()
            return obj
    except ImportError:
        pass
    
    if isinstance(obj, dict):
        # Check if this is a dict subclass from a custom module
        dict_class = type(obj)
        dict_module = getattr(dict_class, '__module__', '') or ''
        if dict_class is not dict and dict_module and not dict_module.startswith(('builtins', 'comfy', 'pyisolate', 'torch', 'numpy', 'typing')):
            # Custom dict subclass - convert to plain dict
            logger.warning(
                f"📚 [PyIsolate][Serialization] Converting {dict_class.__name__} from {dict_module} to plain dict"
            )
            return {k: _tensor_to_cpu(v) for k, v in obj.items()}
        return {k: _tensor_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Check if this is a list/tuple subclass from a custom module
        seq_class = type(obj)
        seq_module = getattr(seq_class, '__module__', '') or ''
        if seq_class not in (list, tuple) and seq_module and not seq_module.startswith(('builtins', 'comfy', 'pyisolate', 'torch', 'numpy', 'typing')):
            # Custom sequence subclass - convert to plain list/tuple
            logger.warning(
                f"📚 [PyIsolate][Serialization] Converting {seq_class.__name__} from {seq_module} to plain sequence"
            )
        converted = [_tensor_to_cpu(item) for item in obj]
        return tuple(converted) if isinstance(obj, tuple) else converted
    
    # Safety check: primitives pass through, complex objects need pickle validation
    if isinstance(obj, (str, int, float, bool, type(None), bytes)):
        return obj
    
    # Check if this object references a module that won't exist on the host
    # Custom node modules (like ComfyUI_Lora_Manager) aren't available on host
    obj_module = getattr(type(obj), '__module__', '') or ''
    if obj_module and not obj_module.startswith(('builtins', 'comfy', 'pyisolate', 'torch', 'numpy', 'typing', 'collections', 'abc')):
        # This is likely a custom node object - convert to serializable form
        type_name = type(obj).__name__
        logger.warning(
            f"📚 [PyIsolate][Serialization] Converting {type_name} from {obj_module} to string (pickle safety)"
        )
        return {
            "__pyisolate_unpicklable__": True,
            "type": type_name,
            "module": obj_module,
            "repr": str(obj)[:1000],  # Limit repr length
        }
    
    return obj


def _tensor_to_cuda(obj: Any, device: Any = None) -> Any:
    """
    Previously converted CPU tensors back to CUDA, but this causes issues because
    ComfyUI expects image tensors to remain on CPU. Now this is a no-op passthrough.
    Tensors stay on whatever device they were after CPU conversion for serialization.
    
    Also handles custom object deserialization (e.g., ModelPatcher for ComfyUI).
    """
    # Custom deserialization hook for ModelPatcherRef (ComfyUI integration)
    if isinstance(obj, dict) and obj.get("__type__") == "ModelPatcherRef":
        try:
            import os
            is_child = os.environ.get("PYISOLATE_CHILD") == "1"
            
            if is_child:
                # Child-side: deserialize ref to ModelPatcherProxy
                from comfy.isolation.model_patcher_proxy import ModelPatcherProxy
                
                model_id = obj["model_id"]
                logger.info(f"📚 [PyIsolate][Deserialization] ref {model_id} → ModelPatcherProxy (child)")
                # Child-side proxy - registry resolved dynamically, no lifecycle management
                return ModelPatcherProxy(model_id, registry=None, manage_lifecycle=False)
            else:
                # Host-side: deserialize ref back to real ModelPatcher via registry
                from comfy.isolation.model_patcher_proxy import ModelPatcherRegistry
                registry = ModelPatcherRegistry()
                model_id = obj["model_id"]
                patcher = registry._get_instance(model_id)
                logger.debug(f"📚 [PyIsolate][Deserialization] ref {model_id} → ModelPatcher (host)")
                return patcher
        except ImportError as e:
            logger.debug(f"📚 [PyIsolate][Deserialization] ComfyUI integration not available: {e}")
        except ValueError as e:
            logger.warning(f"📚 [PyIsolate][Deserialization] {e}")
    
    if isinstance(obj, dict):
        return {k: _tensor_to_cuda(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        converted = [_tensor_to_cuda(item, device) for item in obj]
        return type(obj)(converted) if isinstance(obj, tuple) else converted
    
    # No-op: don't move tensors to CUDA, leave them on CPU as ComfyUI expects
    return obj


def local_execution(func):
    """Decorator to mark a ProxiedSingleton method for local execution.

    By default, all methods in a ProxiedSingleton are executed on the host
    process via RPC. Use this decorator to mark methods that should run
    locally in each process instead.

    This is useful for methods that:
    - Need to access process-local state (e.g., caches, metrics)
    - Don't need to be synchronized across processes
    - Would have poor performance if executed via RPC

    Args:
        func: The method to mark for local execution.

    Returns:
        The decorated method that will execute locally.

    Example:
        >>> class CachedService(ProxiedSingleton):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self._local_cache = {}
        ...         self.shared_data = {}
        ...
        ...     async def get_shared(self, key: str) -> Any:
        ...         # This runs on the host via RPC
        ...         return self.shared_data.get(key)
        ...
        ...     @local_execution
        ...     def get_cache_size(self) -> int:
        ...         # This runs locally in each process
        ...         return len(self._local_cache)

    Note:
        Local methods can be synchronous or asynchronous, but they cannot
        access shared state from the host process.
    """
    func._is_local_execution = True
    return func


class LocalMethodRegistry:
    """Registry for local method implementations in proxied singletons"""

    _instance: LocalMethodRegistry | None = None
    _lock = threading.Lock()

    def __init__(self):
        self._local_implementations: dict[type, object] = {}
        self._local_methods: dict[type, set[str]] = {}

    @classmethod
    def get_instance(cls) -> LocalMethodRegistry:
        """Get the singleton instance of LocalMethodRegistry"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register_class(self, cls: type) -> None:
        """Register a class with its local method implementations"""
        # Create a local instance by bypassing the singleton mechanism
        # We call the base object.__new__ directly to avoid getting the existing singleton
        local_instance = object.__new__(cls)  # type: ignore[misc]
        cls.__init__(local_instance)
        self._local_implementations[cls] = local_instance

        # Track which methods are marked for local execution
        local_methods = set()
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if getattr(method, "_is_local_execution", False):
                local_methods.add(name)

        # Also check instance methods
        for name in dir(cls):
            if not name.startswith("_"):
                attr = getattr(cls, name, None)
                if callable(attr) and getattr(attr, "_is_local_execution", False):
                    local_methods.add(name)

        self._local_methods[cls] = local_methods

    def is_local_method(self, cls: type, method_name: str) -> bool:
        """Check if a method should be executed locally"""
        return cls in self._local_methods and method_name in self._local_methods[cls]

    def get_local_method(self, cls: type, method_name: str):
        """Get the local implementation of a method"""
        if cls not in self._local_implementations:
            raise ValueError(f"Class {cls} not registered for local execution")

        local_instance = self._local_implementations[cls]
        return getattr(local_instance, method_name)


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


RPCPendingMessage = Union[RPCPendingRequest, RPCResponse]


proxied_type = TypeVar("proxied_type", bound=object)


current_rpc_context: contextvars.ContextVar[AsyncRPC | None] = contextvars.ContextVar("current_rpc_context", default=None)


class AsyncRPC:
    def __init__(
        self,
        recv_queue: typehint_mp.Queue[RPCMessage],
        send_queue: typehint_mp.Queue[RPCMessage],
    ):
        self.id = str(uuid.uuid4())
        self.handling_call_id: contextvars.ContextVar[int | None]
        self.handling_call_id = contextvars.ContextVar(self.id + "_handling_call_id", default=None)
        self.recv_queue = recv_queue
        self.send_queue = send_queue
        self.lock = threading.Lock()
        self.pending: dict[int, RPCPendingRequest] = {}
        self.default_loop = asyncio.get_event_loop()
        self.callees: dict[str, object] = {}
        self.callbacks: dict[str, Any] = {}
        self.blocking_future: asyncio.Future | None = None

        # Use an outbox to avoid blocking when we try to send
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
        result = await pending_request["future"]
        return result

    def create_caller(self, abc: type[proxied_type], object_id: str) -> proxied_type:
        this = self

        class CallWrapper:
            def __init__(self):
                pass

            def __getattr__(self, name):
                attr = getattr(abc, name, None)
                if not callable(attr) or name.startswith("_"):
                    raise AttributeError(f"{name} is not a valid method")

                # Check if this method should run locally
                registry = LocalMethodRegistry.get_instance()
                if registry.is_local_method(abc, name):
                    return registry.get_local_method(abc, name)

                # Original RPC logic for remote methods
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
                    result = await pending_request["future"]
                    return result

                return method

        return cast(proxied_type, CallWrapper())

    def register_callee(self, object_instance: object, object_id: str):
        with self.lock:
            if object_id in self.callees:
                raise ValueError(f"Object ID {object_id} already registered")
            self.callees[object_id] = object_instance

    async def run_until_stopped(self):
        # Start the threads
        if self.blocking_future is None:
            self.run()
        assert self.blocking_future is not None, "RPC must be running to wait"
        await self.blocking_future

    async def stop(self):
        # Stop the threads by sending None to the queues
        assert self.blocking_future is not None, "RPC must be running to stop"
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
                callback_id = request["callback_id"]
                args = request["args"]
                kwargs = request["kwargs"]

                callback = None
                with self.lock:
                    callback = self.callbacks.get(callback_id)

                if callback is None:
                    raise ValueError(f"Callback ID {callback_id} not found")

                debugprint("Dispatching callback: ", request)
                result = (
                    (await callback(*args, **kwargs))
                    if inspect.iscoroutinefunction(callback)
                    else callback(*args, **kwargs)
                )
            else:
                object_id = request["object_id"]
                method = request["method"]
                args = request["args"]
                kwargs = request["kwargs"]

                callee = None
                with self.lock:
                    callee = self.callees.get(object_id, None)

                if callee is None:
                    raise ValueError(f"Object ID {object_id} not registered for remote calls")

                # Call the method on the callee
                debugprint("Dispatching request: ", request)
                func = getattr(callee, method)
                result = (
                    (await func(*args, **kwargs))
                    if inspect.iscoroutinefunction(func)
                    else func(*args, **kwargs)
                )
            response = RPCResponse(
                kind="response",
                call_id=request["call_id"],
                result=result,
                error=None,
            )
        except Exception as exc:
            error_msg = str(exc)
            logger.exception(
                "📚 [PyIsolate][RPC] Dispatch failed for %s: %s",
                request.get("object_id", request.get("callback_id", "unknown")),
                error_msg,
            )
            response = RPCResponse(
                kind="response",
                call_id=request["call_id"],
                result=None,
                error=error_msg,
            )
        finally:
            current_rpc_context.reset(token)

        debugprint("Sending response: ", response)
        try:
            # Convert CUDA tensors to CPU before sending to avoid cudaMallocAsync IPC issues
            safe_response = _tensor_to_cpu(response)
            self.send_queue.put(safe_response)
        except Exception as exc:
            message = f"📚 [PyIsolate][RPC] Failed sending response (rpc_id={self.id}): {exc}"
            logger.exception(message)
            raise RuntimeError(message) from exc

    def _recv_thread(self):
        while True:
            try:
                item = self.recv_queue.get()
                # Convert CPU tensors back to CUDA after receiving
                item = _tensor_to_cuda(item)
            except Exception as exc:
                message = f"📚 [PyIsolate][RPC] Failed receiving message (rpc_id={self.id}): {exc}"
                logger.exception(message)
                raise RuntimeError(message) from exc

            debugprint("Got recv: ", item)
            if item is None:
                if self.blocking_future:
                    self.default_loop.call_soon_threadsafe(self.blocking_future.set_result, None)
                break

            if item["kind"] == "response":
                debugprint("Got response: ", item)
                call_id = item["call_id"]
                pending_request = None
                with self.lock:
                    pending_request = self.pending.pop(call_id, None)
                debugprint("Pending request: ", pending_request)
                if pending_request:
                    if "error" in item and item["error"] is not None:
                        debugprint("Error in response: ", item["error"])
                        pending_request["calling_loop"].call_soon_threadsafe(
                            pending_request["future"].set_exception,
                            Exception(item["error"]),
                        )
                    else:
                        debugprint("Got result: ", item["result"])
                        set_result = pending_request["future"].set_result
                        result = item["result"]
                        pending_request["calling_loop"].call_soon_threadsafe(set_result, result)
                else:
                    # If we don"t have a pending request, I guess we just continue on
                    continue
            elif item["kind"] == "call" or item["kind"] == "callback":
                request = cast(Union[RPCRequest, RPCCallback], item)
                debugprint("Got call: ", request)
                request_parent = request.get("parent_call_id", None)
                call_id = request["call_id"]

                call_on_loop = self.default_loop
                if request_parent is not None:
                    # Get pending request without holding the lock for long
                    pending_request = None
                    with self.lock:
                        pending_request = self.pending.get(request_parent, None)
                    if pending_request:
                        call_on_loop = pending_request["calling_loop"]

                async def call_with_context(captured_request: RPCRequest | RPCCallback):
                    # Set the context variable directly when the coroutine actually runs
                    token = self.handling_call_id.set(captured_request["call_id"])
                    try:
                        # Run the dispatch directly
                        return await self.dispatch_request(captured_request)
                    finally:
                        # Reset the context variable when done
                        self.handling_call_id.reset(token)

                asyncio.run_coroutine_threadsafe(coro=call_with_context(request), loop=call_on_loop)
            else:
                raise ValueError(f"Unknown item type: {type(item)}")

    def _send_thread(self):
        id_gen = 0
        while True:
            item = self.outbox.get()
            if item is None:
                break

            debugprint("Got send: ", item)
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
                    message = (
                        f"📚 [PyIsolate][RPC] Failed sending RPC request "
                        f"(rpc_id={self.id}, method={item['method']}): {exc}"
                    )
                    with self.lock:
                        pending = self.pending.pop(call_id, None)
                    if pending:
                        pending["calling_loop"].call_soon_threadsafe(
                            pending["future"].set_exception,
                            RuntimeError(message),
                        )
                    logger.exception(message)
                    raise RuntimeError(message) from exc
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
                    message = (
                        f"📚 [PyIsolate][RPC] Failed sending RPC callback "
                        f"(rpc_id={self.id}): {exc}"
                    )
                    with self.lock:
                        pending = self.pending.pop(call_id, None)
                    if pending:
                        pending["calling_loop"].call_soon_threadsafe(
                            pending["future"].set_exception,
                            RuntimeError(message),
                        )
                    logger.exception(message)
                    raise RuntimeError(message) from exc
            elif item["kind"] == "response":
                try:
                    safe_item = _tensor_to_cpu(item)
                    self.send_queue.put(safe_item)
                except Exception as exc:
                    message = f"📚 [PyIsolate][RPC] Failed relaying response (rpc_id={self.id}): {exc}"
                    logger.exception(message)
                    raise RuntimeError(message) from exc
            else:
                raise ValueError(f"Unknown item type: {type(item)}")


class SingletonMetaclass(type):
    T = TypeVar("T", bound="SingletonMetaclass")
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    def inject_instance(cls: type[T], instance: T) -> None:
        assert cls not in SingletonMetaclass._instances, "Cannot inject instance after first instantiation"
        SingletonMetaclass._instances[cls] = instance

    def get_instance(cls: type[T], *args, **kwargs) -> T:
        """
        Gets the singleton instance of the class, creating it if it doesn't exist.
        """
        if cls not in SingletonMetaclass._instances:
            SingletonMetaclass._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    def use_remote(cls, rpc: AsyncRPC) -> None:
        assert issubclass(cls, ProxiedSingleton), (
            "Class must be a subclass of ProxiedSingleton to be made remote"
        )
        id = cls.get_remote_id()
        remote = rpc.create_caller(cls, id)

        # Register local implementations for methods marked with @local_execution
        registry = LocalMethodRegistry.get_instance()
        registry.register_class(cls)

        cls.inject_instance(remote)  # type: ignore

        for name, t in get_type_hints(cls).items():
            if isinstance(t, type) and issubclass(t, ProxiedSingleton) and not name.startswith("_"):
                # If the type is a ProxiedSingleton, we need to register it as well
                assert issubclass(t, ProxiedSingleton), f"{t} must be a subclass of ProxiedObject"
                caller = rpc.create_caller(t, t.get_remote_id())
                setattr(remote, name, caller)


class ProxiedSingleton(metaclass=SingletonMetaclass):
    """Base class for creating shared singleton services across processes.

    ProxiedSingleton enables you to create services that have a single instance
    shared across all extensions and the host process. When an extension accesses
    a ProxiedSingleton, it automatically gets a proxy to the singleton instance
    in the host process, ensuring all processes share the same state.

    This is particularly useful for shared resources like databases, configuration
    managers, or any service that should maintain consistent state across all
    extensions.

    Advanced usage: Methods can be marked to run locally in each process instead
    of being proxied to the host (see internal documentation for details).

    Example:
        >>> from pyisolate import ProxiedSingleton
        >>>
        >>> class DatabaseService(ProxiedSingleton):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.data = {}
        ...
        ...     async def get(self, key: str) -> Any:
        ...         return self.data.get(key)
        ...
        ...     async def set(self, key: str, value: Any) -> None:
        ...         self.data[key] = value
        ...
        >>>
        >>> # In extension configuration:
        >>> config = ExtensionConfig(
        ...     name="my_extension",
        ...     module_path="./extension.py",
        ...     apis=[DatabaseService],  # Grant access to this singleton
        ...     # ... other config
        ... )

    Note:
        All methods that should be accessible via RPC must be async methods.
        Synchronous methods can only be used if marked with @local_execution.
    """

    def __init__(self):
        """Initialize the ProxiedSingleton.

        This constructor is called only once per singleton class in the host
        process. Extensions will receive a proxy instead of creating new instances.
        """
        super().__init__()

    @classmethod
    def get_remote_id(cls) -> str:
        """Get the unique identifier for this singleton in the RPC system.

        By default, this returns the class name. Override this method if you
        need a different identifier (e.g., to avoid naming conflicts).

        You probably don't need to override this.

        Returns:
            The string identifier used to register and look up this singleton
            in the RPC system.
        """
        return cls.__name__

    def _register(self, rpc: AsyncRPC):
        """Register this singleton instance with the RPC system.

        This method is called automatically by the framework to make this
        singleton available for remote calls. It should not be called directly
        by user code.

        Args:
            rpc: The AsyncRPC instance to register with.
        """
        id = self.get_remote_id()
        rpc.register_callee(self, id)

        # Iterate through all attributes on the class and register any that are also ProxiedSingleton
        for name, attr in self.__class__.__dict__.items():
            if isinstance(attr, ProxiedSingleton) and not name.startswith("_"):
                # Prevent infinite recursion if the attribute is self (e.g. Singleton.instance = self)
                if attr is self:
                    continue
                attr._register(rpc)
