import hashlib
import logging
import os
import socket
import subprocess
import sys
import tempfile
import threading
from logging.handlers import QueueListener
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

from ..config import ExtensionConfig
from ..shared import ExtensionBase
from .environment import (
    build_extension_snapshot,
    create_venv,
    install_dependencies,
    normalize_extension_name,
    validate_dependency,
    validate_path_within_root,
)
from .rpc_protocol import AsyncRPC
from .rpc_transports import JSONSocketTransport
from .tensor_serializer import register_tensor_serializer
from .torch_utils import probe_cuda_ipc_support

__all__ = [
    "Extension",
    "ExtensionBase",
    "build_extension_snapshot",
    "normalize_extension_name",
    "validate_dependency",
]

logger = logging.getLogger(__name__)


class _DeduplicationFilter(logging.Filter):
    def __init__(self, timeout_seconds: int = 10):
        super().__init__()
        self.timeout = timeout_seconds
        self.last_seen: dict[str, float] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        import time
        msg_content = record.getMessage()
        msg_hash = hashlib.sha256(msg_content.encode('utf-8')).hexdigest()
        now = time.time()

        if msg_hash in self.last_seen and now - self.last_seen[msg_hash] < self.timeout:
            return False  # Suppress duplicate

        self.last_seen[msg_hash] = now

        if len(self.last_seen) > 1000:
            cutoff = now - self.timeout
            self.last_seen = {k: v for k, v in self.last_seen.items() if v > cutoff}

        return True


T = TypeVar("T", bound=ExtensionBase)


class Extension(Generic[T]):
    def __init__(
        self,
        module_path: str,
        extension_type: type[T],
        config: ExtensionConfig,
        venv_root_path: str,
    ) -> None:
        force_ipc = os.environ.get("PYISOLATE_FORCE_CUDA_IPC") == "1"

        if "share_cuda_ipc" not in config:
            config["share_cuda_ipc"] = force_ipc
        elif force_ipc:
            config["share_cuda_ipc"] = True

        self.name = config["name"]
        self.normalized_name = normalize_extension_name(self.name)

        for dep in config["dependencies"]:
            validate_dependency(dep)

        venv_root = Path(venv_root_path).resolve()
        self.venv_path = venv_root / self.normalized_name
        validate_path_within_root(self.venv_path, venv_root)

        self.module_path = module_path
        self.config = config
        self.extension_type = extension_type
        self._cuda_ipc_enabled = False

        # Auto-populate APIs from adapter if not already in config
        if "apis" not in self.config:
            try:
                # v1.0: Check registry
                from .adapter_registry import AdapterRegistry
                adapter = AdapterRegistry.get()

                if adapter:
                    rpc_services = adapter.provide_rpc_services()
                    self.config["apis"] = rpc_services
                    logger.info("[Extension] Auto-populated %d RPC services from adapter", len(rpc_services))
                else:
                    self.config["apis"] = []
            except Exception as exc:
                logger.warning("[Extension] Could not load adapter RPC services: %s", exc)
                self.config["apis"] = []

        self.mp: Any
        if self.config["share_torch"]:
            import torch.multiprocessing
            self.mp = torch.multiprocessing
        else:
            import multiprocessing
            self.mp = multiprocessing

        self._process_initialized = False
        self.log_queue: Optional[Any] = None
        self.log_listener: Optional[QueueListener] = None

        # UDS / JSON-RPC resources
        self._uds_listener: Optional[Any] = None
        self._uds_path: Optional[str] = None
        self._client_sock: Optional[Any] = None

        self.extension_proxy: Optional[T] = None

    def ensure_process_started(self) -> None:
        """Start the isolated process if it has not been initialized."""
        if self._process_initialized:
            return
        self._initialize_process()
        self._process_initialized = True

    def _initialize_process(self) -> None:
        """Initialize queues, RPC, and launch the isolated process."""
        try:
            self.ctx = self.mp.get_context("spawn")
        except ValueError as e:
            raise RuntimeError(f"Failed to get 'spawn' context: {e}") from e

        # Determine CUDA IPC eligibility up front (host side)
        self._cuda_ipc_enabled = False
        want_ipc = bool(self.config.get("share_cuda_ipc", False))
        if want_ipc:
            if not self.config.get("share_torch", False):
                raise RuntimeError("share_cuda_ipc requires share_torch=True")
            supported, reason = probe_cuda_ipc_support()
            if not supported:
                raise RuntimeError(f"CUDA IPC requested but unavailable: {reason}")
            self._cuda_ipc_enabled = True
            logger.debug("CUDA IPC enabled for %s", self.name)

        os.environ["PYISOLATE_ENABLE_CUDA_IPC"] = "1" if self._cuda_ipc_enabled else "0"

        # NOTE: PYISOLATE_CHILD is set in the child's env dict, NOT in os.environ
        # Setting it in os.environ would affect the HOST process serialization logic

        if os.name == "nt":
            import multiprocessing as std_mp
            std_ctx = std_mp.get_context("spawn")
            self.log_queue = std_ctx.Manager().Queue()
        else:
            self.log_queue = self.ctx.Queue()

        self.extension_proxy = None

        # Create handler with deduplication filter (industry standard)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.addFilter(_DeduplicationFilter(timeout_seconds=5))

        self.log_listener = QueueListener(self.log_queue, stream_handler)
        self.log_listener.start()

        # Register tensor serializer for JSON-RPC
        from .serialization_registry import SerializerRegistry
        register_tensor_serializer(SerializerRegistry.get_instance())

        # Ensure file_system strategy for CPU tensors
        import torch
        torch.multiprocessing.set_sharing_strategy('file_system')

        self.proc = self.__launch()

        for api in self.config["apis"]:
            api()._register(self.rpc)

        self.rpc.run()

    def get_proxy(self) -> T:
        """Return (and memoize) the RPC caller for the remote extension."""
        if self.extension_proxy is None:
            self.extension_proxy = self.rpc.create_caller(self.extension_type, "extension")
        return self.extension_proxy

    def stop(self) -> None:
        """Stop the extension process and clean up queues/listeners."""
        errors: list[str] = []

        # Terminate process
        if hasattr(self, "proc") and self.proc:
            try:
                if self.proc.poll() is None:
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
                        self.proc.wait()
            except Exception as exc:
                errors.append(f"terminate: {exc}")

        if self.log_listener:
            try:
                self.log_listener.stop()
            except Exception as exc:
                errors.append(f"log_listener: {exc}")

        # Clean up UDS resources
        if self._client_sock:
            try:
                self._client_sock.close()
            except Exception as exc:
                errors.append(f"client_sock: {exc}")

        if self._uds_listener:
            try:
                self._uds_listener.close()
            except Exception as exc:
                errors.append(f"uds_listener: {exc}")

        if self._uds_path and os.path.exists(self._uds_path):
            try:
                os.unlink(self._uds_path)
            except Exception as exc:
                errors.append(f"unlink uds: {exc}")

        if self.log_queue:
            try:
                # On Windows/multiprocessing, queue might need closing
                if hasattr(self.log_queue, "close"):
                    self.log_queue.close()
            except Exception as exc:
                errors.append(f"log_queue: {exc}")

        self._process_initialized = False
        self.extension_proxy = None
        if hasattr(self, "rpc"):
            del self.rpc

        if errors:
            raise RuntimeError(f"Errors stopping {self.name}: {'; '.join(errors)}")

    def __launch(self) -> Any:
        """Launch the extension in a separate process after venv + deps are ready."""
        create_venv(self.venv_path, self.config)
        install_dependencies(self.venv_path, self.config, self.name)
        return self._launch_with_uds()

    def _launch_with_uds(self) -> Any:
        """Launch the extension using UDS + JSON-RPC (Standard Isolation)."""
        import os as os_module

        # Determine Python executable
        if os.name == "nt":
            python_exe = str(self.venv_path / "Scripts" / "python.exe")
        else:
            python_exe = str(self.venv_path / "bin" / "python")

        # Create UDS listener
        uid = os_module.getuid()
        run_dir = f"/run/user/{uid}/pyisolate"
        if not os_module.path.exists(run_dir):
            try:
                os_module.makedirs(run_dir, mode=0o700)
            except PermissionError:
                run_dir = f"/tmp/pyisolate-{uid}"
                os_module.makedirs(run_dir, mode=0o700, exist_ok=True)

        uds_path = tempfile.mktemp(prefix="ext_", suffix=".sock", dir=run_dir)

        listener_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener_sock.bind(uds_path)
        os_module.chmod(uds_path, 0o600)
        listener_sock.listen(1)

        self._uds_listener = listener_sock
        self._uds_path = uds_path

        logger.info("[PyIsolate][JSON-RPC] Listening on %s", uds_path)

        # Build command
        cmd = [python_exe, "-m", "pyisolate._internal.uds_client"]

        # Prepare environment
        env = os.environ.copy()
        env["PYISOLATE_UDS_ADDRESS"] = uds_path
        env["PYISOLATE_CHILD"] = "1"
        env["PYISOLATE_EXTENSION"] = self.name
        env["PYISOLATE_MODULE_PATH"] = self.module_path
        env["PYISOLATE_ENABLE_CUDA_IPC"] = "1" if self._cuda_ipc_enabled else "0"

        # Launch process
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=None, # Inherit stdout/stderr for now so we see logs
            stderr=None,
            close_fds=True,
        )

        # Accept connection
        client_sock = None
        accept_error = None

        def accept_connection() -> None:
            nonlocal client_sock, accept_error
            try:
                client_sock, _ = listener_sock.accept()
            except Exception as e:
                accept_error = e

        accept_thread = threading.Thread(target=accept_connection)
        accept_thread.daemon = True
        accept_thread.start()
        accept_thread.join(timeout=30.0)

        if accept_thread.is_alive():
            proc.terminate()
            raise RuntimeError(f"Child failed to connect within timeout for {self.name}")

        if accept_error:
            proc.terminate()
            raise RuntimeError(f"Child failed to connect for {self.name}: {accept_error}")

        if client_sock is None:
            proc.terminate()
            raise RuntimeError(f"Child connection is None for {self.name}")

        # Setup JSON-RPC
        transport = JSONSocketTransport(client_sock)
        logger.debug("Child connected, sending bootstrap data")

        # Send bootstrap
        snapshot = build_extension_snapshot(self.module_path)
        ext_type_ref = f"{self.extension_type.__module__}.{self.extension_type.__name__}"

        # Sanitize config for JSON serialization (convert API classes to string refs)
        safe_config = dict(self.config)  # type: ignore[arg-type]
        if "apis" in safe_config:
            api_list: list[str] = [
                f"{api.__module__}.{api.__name__}" for api in self.config["apis"]  # type: ignore[union-attr]
            ]
            safe_config["apis"] = api_list

        bootstrap_data = {
            "snapshot": snapshot,
            "config": safe_config,
            "extension_type_ref": ext_type_ref,
        }
        transport.send(bootstrap_data)

        self._client_sock = client_sock
        self.rpc = AsyncRPC(transport=transport)

        return proc

    def join(self) -> None:
        """Join the child process, blocking until it exits."""
        self.proc.join()
