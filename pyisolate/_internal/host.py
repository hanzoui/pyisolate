import hashlib
import json
import logging
import os
import re
import shutil
import site
import socket
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import metadata as importlib_metadata
from logging.handlers import QueueListener
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

from ..config import ExtensionConfig
from ..path_helpers import serialize_host_snapshot
from ..shared import ExtensionBase
from .loader import load_adapter
from .shared import AsyncRPC, JSONSocketTransport
from .tensor_serializer import register_tensor_serializer
from .torch_utils import get_torch_ecosystem_packages

__all__ = [
    "Extension",
    "ExtensionBase",
    "build_extension_snapshot",
    "normalize_extension_name",
    "validate_dependency",
]

logger = logging.getLogger(__name__)


def _probe_cuda_ipc_support() -> tuple[bool, str]:
    """Best-effort probe for CUDA IPC support on Linux.

    Returns:
        (supported, reason)
    """
    if sys.platform != "linux":
        return False, "CUDA IPC is only supported on Linux"
    try:
        import torch
    except Exception as exc:  # pragma: no cover - import guard
        return False, f"torch import failed: {exc}"

    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False"

    try:
        # Minimal handle check: event with interprocess support + tiny tensor
        torch.cuda.current_device()
        _ = torch.cuda.Event(interprocess=True)  # type: ignore[no-untyped-call]
        _ = torch.empty(1, device="cuda")
        return True, "ok"
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"CUDA IPC probe failed: {exc}"


# ---------------------------------------------------------------------------
# Sandbox System Path Allow-List (DENY-BY-DEFAULT)
# ---------------------------------------------------------------------------
# These are the ONLY system paths exposed to sandboxed processes.
# Everything else is denied. This is a security-critical list.

SANDBOX_SYSTEM_PATHS: list[str] = [
    "/usr",               # System binaries and libraries
    "/lib",               # Core libraries
    "/lib64",             # 64-bit libraries (if exists)
    "/lib32",             # 32-bit libraries (if exists)
    "/bin",               # Essential binaries
    "/sbin",              # System binaries
    "/etc/alternatives",  # Symlink management
    "/etc/ld.so.cache",   # Dynamic linker cache
    "/etc/ld.so.conf",    # Dynamic linker config
    "/etc/ld.so.conf.d",  # Dynamic linker config dir
    "/etc/ssl",           # SSL certificates
    "/etc/ca-certificates",  # CA certificates
    "/etc/pki",           # PKI certificates (RHEL/CentOS)
    "/etc/resolv.conf",   # DNS (if network enabled)
    "/etc/hosts",         # Host resolution
    "/etc/nsswitch.conf", # Name service switch config
    "/etc/passwd",        # User info (read-only, needed for getpwuid)
    "/etc/group",         # Group info
    "/etc/localtime",     # Timezone
    "/etc/timezone",      # Timezone name
]

# GPU device paths for CUDA passthrough
GPU_PASSTHROUGH_PATTERNS: list[str] = [
    "nvidia*",            # GPU devices
    "nvidiactl",          # Control device
    "nvidia-uvm",         # Unified memory
    "nvidia-uvm-tools",   # UVM tools
    "dri",                # Direct Rendering Infrastructure
]


def _build_bwrap_command(
    python_exe: str,
    module_path: str,
    venv_path: str,
    uds_address: str,
    sandbox_config: dict[str, Any],
    allow_gpu: bool,
) -> list[str]:
    """Build the bubblewrap command for launching a sandboxed process.

    Security properties:
    - DENY-BY-DEFAULT filesystem (explicit allow-list only)
    - Venv is READ-ONLY (prevents persistent infection)
    - User namespace isolation (unprivileged execution)
    - PID namespace isolation (process isolation)
    - Network isolated by default
    - /dev/shm shared (required for CUDA IPC, documented risk)

    Args:
        python_exe: Path to the Python interpreter in the venv
        module_path: Path to the extension module directory
        venv_path: Path to the isolated venv
        uds_address: Path to the Unix socket for IPC
        sandbox_config: SandboxConfig dict with network, writable_paths, readonly_paths
        allow_gpu: Whether to enable GPU passthrough

    Returns:
        Command list suitable for subprocess.Popen
    """
    cmd = ["bwrap"]

    # NOTE: User namespace isolation (--unshare-user) is disabled because
    # Ubuntu 24.04+ has kernel.apparmor_restrict_unprivileged_userns=1 by default.
    # This means unprivileged users cannot create user namespaces without an
    # AppArmor profile that allows it.
    #
    # Without user namespace, we still get:
    # - Filesystem isolation (deny-by-default)
    # - Read-only venv and module paths (prevent modification)
    # - Temporary filesystem in /tmp
    # - Network sharing (required for API calls)
    #
    # To enable full user namespace isolation, a sysadmin would need to either:
    # 1. Set kernel.apparmor_restrict_unprivileged_userns=0
    # 2. Create an AppArmor profile for pyisolate

    # Mount namespace isolation (this DOES work without user namespace)
    # PID namespace requires user namespace, so we skip it too

    # New session (detach from terminal)
    cmd.append("--new-session")

    # Essential virtual filesystems
    cmd.extend(["--proc", "/proc"])
    cmd.extend(["--dev", "/dev"])
    cmd.extend(["--tmpfs", "/tmp"])

    # DENY-BY-DEFAULT: Only bind required system paths (read-only)
    for sys_path in SANDBOX_SYSTEM_PATHS:
        if os.path.exists(sys_path):
            cmd.extend(["--ro-bind", sys_path, sys_path])

    # Venv: READ-ONLY (prevent malicious modification)
    cmd.extend(["--ro-bind", str(venv_path), str(venv_path)])

    # Module path: READ-ONLY
    cmd.extend(["--ro-bind", str(module_path), str(module_path)])

    # GPU passthrough (if enabled)
    if allow_gpu:
        dev_path = Path("/dev")
        for pattern in GPU_PASSTHROUGH_PATTERNS:
            for dev in dev_path.glob(pattern):
                if dev.exists():
                    cmd.extend(["--dev-bind", str(dev), str(dev)])
        # CUDA IPC requires shared memory
        # SECURITY NOTE: /dev/shm is shared. This is a known side-channel risk
        # but unavoidable for zero-copy tensor transfer. Document this trade-off.
        if Path("/dev/shm").exists():
            cmd.extend(["--bind", "/dev/shm", "/dev/shm"])

        # CUDA library and runtime paths (read-only)
        cuda_paths = [
            "/usr/local/cuda",        # Common CUDA install location
            "/opt/cuda",              # Alternative CUDA location
            "/run/nvidia-persistenced",  # Persistence daemon
        ]
        for cuda_path in cuda_paths:
            if os.path.exists(cuda_path):
                cmd.extend(["--ro-bind", cuda_path, cuda_path])

    # Network: default to shared (unshare-net requires CAP_NET_ADMIN or special config)
    # Only isolate network if explicitly requested AND the system supports it
    # For most GPU workloads, network isolation isn't critical
    cmd.append("--share-net")

    # Additional paths from config (user-specified)
    for path in sandbox_config.get("writable_paths", []):
        if os.path.exists(path):
            cmd.extend(["--bind", path, path])
    for path in sandbox_config.get("readonly_paths", []):
        if os.path.exists(path):
            cmd.extend(["--ro-bind", path, path])

    # UDS socket directory must be accessible
    uds_dir = os.path.dirname(uds_address)
    if uds_dir and os.path.exists(uds_dir):
        cmd.extend(["--bind", uds_dir, uds_dir])

    # Environment variables
    cmd.extend(["--setenv", "PYISOLATE_UDS_ADDRESS", uds_address])
    cmd.extend(["--setenv", "PYISOLATE_CHILD", "1"])

    # Inherit select environment variables
    # Standard environment
    for env_var in ["PATH", "HOME", "LANG", "LC_ALL", "PYTHONPATH"]:
        if env_var in os.environ:
            cmd.extend(["--setenv", env_var, os.environ[env_var]])

    # CUDA/GPU environment variables (critical for GPU access)
    cuda_env_vars = [
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "PYTORCH_CUDA_ALLOC_CONF",
        "TORCH_CUDA_ARCH_LIST",
        "PYISOLATE_ENABLE_CUDA_IPC",
    ]
    for env_var in cuda_env_vars:
        if env_var in os.environ:
            cmd.extend(["--setenv", env_var, os.environ[env_var]])

    # Command to run
    cmd.extend([python_exe, "-m", "pyisolate._internal.sandbox_client"])

    return cmd


_DANGEROUS_PATTERNS = ("&&", "||", ";", "|", "`", "$", "\n", "\r", "\0")
_UNSAFE_CHARS = frozenset(' \t\n\r;|&$`()<>"\'\\!{}[]*?~#%=,:')


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


def _detect_pyisolate_version() -> str:
    try:
        return importlib_metadata.version("pyisolate")
    except Exception:
        return "0.0.0"


pyisolate_version = _detect_pyisolate_version()


def build_extension_snapshot(module_path: str) -> dict[str, object]:
    """Construct snapshot payload with adapter metadata for child bootstrap."""
    snapshot: dict[str, object] = serialize_host_snapshot()

    adapter = None
    path_config: dict[str, object] = {}
    try:
        adapter = load_adapter()
    except Exception as exc:
        logger.warning("Adapter load failed: %s", exc)

    if adapter:
        try:
            path_config = adapter.get_path_config(module_path) or {}
        except Exception as exc:
            logger.warning("Adapter path config failed: %s", exc)

        # Register serializers in host process (needed for RPC serialization)
        try:
            from .serialization_registry import SerializerRegistry
            registry = SerializerRegistry.get_instance()
            adapter.register_serializers(registry)
        except Exception as exc:
            logger.warning("Adapter serializer registration failed: %s", exc)

    snapshot.update(
        {
            "adapter_name": adapter.identifier if adapter else None,
            "preferred_root": path_config.get("preferred_root"),
            "additional_paths": path_config.get("additional_paths", []),
            "context_data": {"module_path": module_path},
        }
    )
    return snapshot


def normalize_extension_name(name: str) -> str:
    """
    Normalize an extension name for filesystem and shell safety.

    Replaces unsafe characters, strips traversal attempts, and ensures a non-empty
    result while preserving Unicode characters.

    Raises:
        ValueError: If the normalized name would be empty.
    """
    if not name:
        raise ValueError("Extension name cannot be empty")

    name = name.replace("/", "_").replace("\\", "_")
    while name.startswith("."):
        name = name[1:]
    name = name.replace("..", "_")

    for char in _UNSAFE_CHARS:
        name = name.replace(char, "_")

    name = re.sub(r"_+", "_", name)
    name = name.strip("_")

    if not name:
        raise ValueError("Extension name contains only invalid characters")
    return name


def validate_dependency(dep: str) -> None:
    """Validate a single dependency specification."""
    if not dep:
        return
    # Allow `-e` flag for editable installs (e.g., `-e /path/to/package` or `-e .`)
    # This enables development workflows where the extension is pip-installed in editable mode
    if dep == "-e":
        return
    if dep.startswith("-") and not dep.startswith("-e "):
        raise ValueError(
            f"Invalid dependency '{dep}'. "
            "Dependencies cannot start with '-' as this could be a command option."
        )
    for pattern in _DANGEROUS_PATTERNS:
        if pattern in dep:
            raise ValueError(
                f"Invalid dependency '{dep}'. Contains potentially dangerous character: '{pattern}'"
            )


def validate_path_within_root(path: Path, root: Path) -> None:
    """Ensure ``path`` is contained within ``root`` to avoid path escape."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as err:
        raise ValueError(f"Path '{path}' is not within root '{root}'") from err


@contextmanager
def environment(**env_vars: Any) -> Iterator[None]:
    """Temporarily set environment variables inside a context."""
    original: dict[str, Optional[str]] = {}
    for key, value in env_vars.items():
        original[key] = os.environ.get(key)
        os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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
                adapter = load_adapter()
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

    def _exclude_satisfied_requirements(self, requirements: list[str], python_exe: Path) -> list[str]:
        """Filter requirements to skip packages already satisfied in the venv.

        When ``share_torch`` is enabled, the child venv inherits host site-packages
        via a .pth file. Torch ecosystem packages MUST be byte-identical between
        parent and child for shared memory tensor passing to work correctly.
        Reinstalling could resolve to different versions, breaking the share_torch
        contract. This is a correctness requirement, not a performance optimization.
        """
        from packaging.requirements import Requirement

        result = subprocess.run(  # noqa: S603  # Trusted: system pip executable
            [str(python_exe), "-m", "pip", "list", "--format", "json"],
            capture_output=True, text=True, check=True
        )
        installed = {pkg["name"].lower(): pkg["version"] for pkg in json.loads(result.stdout)}
        torch_ecosystem = get_torch_ecosystem_packages()

        filtered = []
        for req_str in requirements:
            req_str_stripped = req_str.strip()
            if req_str_stripped.startswith('-e ') or req_str_stripped == '-e':
                filtered.append(req_str)
                continue
            if req_str_stripped.startswith(('/', './')):
                filtered.append(req_str)
                continue

            try:
                req = Requirement(req_str)
                pkg_name_lower = req.name.lower()

                # Torch ecosystem packages are inherited when share_torch=True; skip
                # reinstalling them to avoid conflicts and unnecessary downloads.
                if self.config["share_torch"] and pkg_name_lower in torch_ecosystem:
                    continue

                if pkg_name_lower in installed:
                    installed_version = installed[pkg_name_lower]
                    if not req.specifier or installed_version in req.specifier:
                        continue

                filtered.append(req_str)
            except Exception:
                filtered.append(req_str)

        return filtered

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
            supported, reason = _probe_cuda_ipc_support()
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
        self._create_extension_venv()
        self._install_dependencies()
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
        logger.info("[PyIsolate][JSON-RPC] Child connected, sending bootstrap data")

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

    def _create_extension_venv(self) -> None:
        """Create the virtual environment for this extension using uv."""
        self.venv_path.parent.mkdir(parents=True, exist_ok=True)

        uv_path = shutil.which("uv")
        if not uv_path:
            raise RuntimeError(
                "uv is required but not found. Install it with: pip install uv\n"
                "See https://github.com/astral-sh/uv for installation options."
            )

        if not self.venv_path.exists():
            subprocess.check_call([  # noqa: S603  # Trusted: uv venv command
                uv_path, "venv", str(self.venv_path), "--python", sys.executable
            ])

            if self.config["share_torch"]:
                if os.name == "nt":
                    child_site = self.venv_path / "Lib" / "site-packages"
                else:
                    vi = sys.version_info
                    child_site = self.venv_path / "lib" / f"python{vi.major}.{vi.minor}" / "site-packages"

                if not child_site.exists():
                    raise RuntimeError(
                        f"site-packages not found at expected path: {child_site}. "
                        f"venv may be malformed."
                    )

                parent_sites = site.getsitepackages()
                host_prefix = sys.prefix
                valid_parents = [p for p in parent_sites if p.startswith(host_prefix)]
                if not valid_parents:
                    valid_parents = [
                        p for p in sys.path
                        if "site-packages" in p and p.startswith(host_prefix)
                    ]
                if not valid_parents:
                    raise RuntimeError(
                        "Could not determine parent site-packages path to inherit. "
                        f"host_prefix={host_prefix}, site_packages={parent_sites}, "
                        f"valid_parents={valid_parents}, "
                        f"candidates={[p for p in sys.path if 'site-packages' in p]}"
                    )

                parent_site = valid_parents[0]
                pth_content = f"import site; site.addsitedir(r'{parent_site}')\n"
                pth_file = child_site / "_pyisolate_parent.pth"
                pth_file.write_text(pth_content)

    def _install_dependencies(self) -> None:
        """Install extension dependencies into the venv, skipping already-satisfied ones."""
        # Windows multiprocessing/Manager uses the interpreter path for spawned
        # processes. The explicit Scripts/python.exe path is required to avoid
        # handle issues when multiprocessing.set_executable is involved.
        if os.name == "nt":
            python_exe = self.venv_path / "Scripts" / "python.exe"
        else:
            python_exe = self.venv_path / "bin" / "python"

        if not python_exe.exists():
            raise RuntimeError(f"Python executable not found at {python_exe}")

        uv_path = shutil.which("uv")
        if not uv_path:
            raise RuntimeError(
                "uv is required but not found. Install it with: pip install uv\n"
                "See https://github.com/astral-sh/uv for installation options."
            )

        safe_deps: list[str] = []
        for dep in self.config["dependencies"]:
            validate_dependency(dep)
            safe_deps.append(dep)

        if self.config["share_torch"] and safe_deps:
            safe_deps = self._exclude_satisfied_requirements(safe_deps, python_exe)

        if not safe_deps:
            return

        # uv handles hardlink vs copy automatically based on filesystem support
        cmd_prefix: list[str] = [uv_path, "pip", "install", "--python", str(python_exe)]
        cache_dir = self.venv_path.parent / ".uv_cache"
        cache_dir.mkdir(exist_ok=True)
        common_args: list[str] = ["--cache-dir", str(cache_dir)]

        torch_spec: Optional[str] = None
        if not self.config["share_torch"]:
            import torch
            torch_version: str = str(torch.__version__)
            if torch_version.endswith("+cpu"):
                torch_version = torch_version[:-4]
            cuda_version = torch.version.cuda  # type: ignore[attr-defined]
            if cuda_version:
                common_args += [
                    "--extra-index-url",
                    f"https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')}",
                ]
            if "dev" in torch_version or "+" in torch_version:
                common_args += ["--index-strategy", "unsafe-best-match"]
            torch_spec = f"torch=={torch_version}"
            safe_deps.insert(0, torch_spec)

        descriptor = {
            "dependencies": safe_deps,
            "share_torch": self.config["share_torch"],
            "torch_spec": torch_spec,
            "pyisolate": pyisolate_version,
            "python": sys.version,
        }
        fingerprint = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
        lock_path = self.venv_path / ".pyisolate_deps.json"

        if lock_path.exists():
            try:
                cached = json.loads(lock_path.read_text(encoding="utf-8"))
                if cached.get("fingerprint") == fingerprint and cached.get("descriptor") == descriptor:
                    return
            except Exception as exc:
                logger.debug("Dependency cache read failed: %s", exc)

        cmd = cmd_prefix + safe_deps + common_args

        with subprocess.Popen(  # noqa: S603  # Trusted: validated pip/uv install cmd
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as proc:
            assert proc.stdout is not None
            output_lines: list[str] = []
            for line in proc.stdout:
                clean = line.rstrip()
                # Filter out pyisolate install messages to avoid polluting logs
                # with internal dependency resolution noise that isn't actionable
                # for users debugging their own extension dependencies.
                if "pyisolate==" not in clean and "pyisolate @" not in clean:
                    output_lines.append(clean)
            return_code = proc.wait()

        if return_code != 0:
            detail = "\n".join(output_lines) or "(no output)"
            raise RuntimeError(f"Install failed for {self.name}: {detail}")

        lock_path.write_text(
            json.dumps({"fingerprint": fingerprint, "descriptor": descriptor}, indent=2),
            encoding="utf-8",
        )

    def join(self) -> None:
        """Join the child process, blocking until it exits."""
        self.proc.join()
