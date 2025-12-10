import hashlib
import json
import logging
import os
import re
import shutil
import site
import subprocess
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from importlib import metadata as importlib_metadata
from logging.handlers import QueueListener
from pathlib import Path
from typing import Generic, Optional, TypeVar

from ..config import ExtensionConfig, get_torch_ecosystem_packages
from ..path_helpers import serialize_host_snapshot
from ..shared import ExtensionBase
from .client import entrypoint
from .shared import AsyncRPC

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
        _ = torch.cuda.Event(interprocess=True)
        _ = torch.empty(1, device="cuda")
        return True, "ok"
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"CUDA IPC probe failed: {exc}"

_DANGEROUS_PATTERNS = ("&&", "||", ";", "|", "`", "$", "\n", "\r", "\0")
_UNSAFE_CHARS = frozenset(' \t\n\r;|&$`()<>"\'\\!{}[]*?~#%=,:')


class _DeduplicationFilter(logging.Filter):
    def __init__(self, timeout_seconds=10):
        super().__init__()
        self.timeout = timeout_seconds
        self.last_seen = {}

    def filter(self, record):
        import time
        msg_content = record.getMessage()
        msg_hash = hashlib.md5(msg_content.encode('utf-8')).hexdigest()
        now = time.time()
        
        if msg_hash in self.last_seen:
            if now - self.last_seen[msg_hash] < self.timeout:
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
def environment(**env_vars):
    """Temporarily set environment variables inside a context."""
    original = {}
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

        if self.config["share_torch"]:
            import torch.multiprocessing
            self.mp = torch.multiprocessing
        else:
            import multiprocessing
            self.mp = multiprocessing

        self._process_initialized = False
        self.log_queue = None
        self.log_listener = None
        self.manager = None

    def ensure_process_started(self) -> None:
        """Start the isolated process if it has not been initialized."""
        if self._process_initialized:
            return
        self._initialize_process()
        self._process_initialized = True

    def _exclude_satisfied_requirements(self, requirements: list[str], python_exe: Path) -> list[str]:
        """Filter requirements to skip packages already satisfied in the venv.

        When ``share_torch`` is enabled, the child venv inherits host site-packages,
        so torch ecosystem packages may already be present. This helper keeps the
        install list minimal.
        """
        from packaging.requirements import Requirement

        result = subprocess.run(
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
            if req_str_stripped.startswith('/') or req_str_stripped.startswith('./'):
                filtered.append(req_str)
                continue

            try:
                req = Requirement(req_str)
                pkg_name_lower = req.name.lower()

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
            logger.info("[PyIsolate][%s] CUDA IPC enabled (share_cuda_ipc=True)", self.name)

        os.environ["PYISOLATE_ENABLE_CUDA_IPC"] = "1" if self._cuda_ipc_enabled else "0"

        os.environ["PYISOLATE_CHILD"] = "1"

        if os.name == "nt":
            import multiprocessing as std_mp
            std_ctx = std_mp.get_context("spawn")
            self.manager = std_ctx.Manager()
            self.to_extension = self.manager.Queue()
            self.from_extension = self.manager.Queue()
            self.log_queue = self.manager.Queue()
        else:
            self.to_extension = self.ctx.Queue()
            self.from_extension = self.ctx.Queue()
            self.log_queue = self.ctx.Queue()

        self.extension_proxy = None
        
        # Create handler with deduplication filter (industry standard)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.addFilter(_DeduplicationFilter(timeout_seconds=5))
        
        self.log_listener = QueueListener(self.log_queue, stream_handler)
        self.log_listener.start()

        self.proc = self.__launch()
        self.rpc = AsyncRPC(recv_queue=self.from_extension, send_queue=self.to_extension)

        for api in self.config["apis"]:
            api()._register(self.rpc)

        self.rpc.run()

    def get_proxy(self) -> T:
        """Return (and memoize) the RPC caller for the remote extension."""
        if self.extension_proxy is None:
            self.extension_proxy = self.rpc.create_caller(self.extension_type, "extension")
        return self.extension_proxy

    def __getattr__(self, name: str):
        """Delegate attribute access to the extension proxy, ensuring the process is running."""
        if name in ("_process_initialized", "ensure_process_started", "get_proxy", "name"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        self.ensure_process_started()
        return getattr(self.get_proxy(), name)

    def stop(self) -> None:
        """Stop the extension process and clean up queues/listeners."""
        errors: list[str] = []

        if hasattr(self, "proc") and self.proc.is_alive():
            try:
                self.proc.terminate()
                self.proc.join(timeout=5.0)
                if self.proc.is_alive():
                    self.proc.kill()
                    self.proc.join()
            except Exception as exc:
                errors.append(f"terminate: {exc}")

        if self.log_listener:
            try:
                self.log_listener.stop()
            except Exception as exc:
                errors.append(f"log_listener: {exc}")

        for attr_name in ("to_extension", "from_extension", "log_queue"):
            q = getattr(self, attr_name, None)
            if q:
                try:
                    q.close()
                except Exception as exc:
                    errors.append(f"{attr_name}: {exc}")

        if self.manager:
            try:
                self.manager.shutdown()
            except Exception as exc:
                errors.append(f"manager: {exc}")

        self._process_initialized = False
        self.extension_proxy = None
        if hasattr(self, "rpc"):
            del self.rpc

        if errors:
            raise RuntimeError(f"Errors stopping {self.name}: {'; '.join(errors)}")

    def __launch(self):
        """Launch the extension in a separate process after venv + deps are ready."""
        self._create_extension_venv()
        self._install_dependencies()

        if os.name == "nt":
            executable = str(self.venv_path / "Scripts" / "python.exe")
        else:
            executable = str(self.venv_path / "bin" / "python")

        snapshot_file = Path(tempfile.gettempdir()) / f"pyisolate_snapshot_{self.name}.json"
        serialize_host_snapshot(output_path=str(snapshot_file))

        self.mp.set_executable(executable)
        with ExitStack() as stack:
            stack.enter_context(
                environment(
                    PYISOLATE_CHILD="1",
                    PYISOLATE_EXTENSION=self.name,
                    PYISOLATE_MODULE_PATH=self.module_path,
                    PYISOLATE_HOST_SNAPSHOT=str(snapshot_file),
                    PYISOLATE_ENABLE_CUDA_IPC="1" if self._cuda_ipc_enabled else "0",
                )
            )
            if os.name == "nt":
                stack.enter_context(environment(VIRTUAL_ENV=str(self.venv_path)))

            proc = self.ctx.Process(
                target=entrypoint,
                args=(
                    self.module_path,
                    self.extension_type,
                    self.config,
                    self.to_extension,
                    self.from_extension,
                    self.log_queue,
                ),
            )
            proc.start()

        return proc

    def _ensure_uv(self) -> bool:
        """Ensure ``uv`` is available; install via pip if missing."""
        if shutil.which("uv"):
            return True
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "uv"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return bool(shutil.which("uv"))
        except Exception:
            return False

    def _create_extension_venv(self):
        """Create the virtual environment for this extension if it does not exist."""
        self.venv_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.venv_path.exists():
            subprocess.check_call([sys.executable, "-m", "venv", str(self.venv_path)])

            if self.config["share_torch"]:
                if os.name == "nt":
                    child_site = self.venv_path / "Lib" / "site-packages"
                else:
                    vi = sys.version_info
                    child_site = self.venv_path / "lib" / f"python{vi.major}.{vi.minor}" / "site-packages"

                if not child_site.exists():
                    found = list(self.venv_path.glob("**/site-packages"))
                    if found:
                        child_site = found[0]
                    else:
                        raise RuntimeError(f"Could not locate site-packages in {self.venv_path}")

                parent_sites = site.getsitepackages()
                host_prefix = sys.prefix
                valid_parents = [p for p in parent_sites if p.startswith(host_prefix)]
                if not valid_parents:
                    valid_parents = [p for p in sys.path if "site-packages" in p and p.startswith(host_prefix)]
                if not valid_parents:
                    raise RuntimeError("Could not determine parent site-packages path")

                parent_site = valid_parents[0]
                pth_content = f"import site; site.addsitedir(r'{parent_site}')\n"
                pth_file = child_site / "_pyisolate_parent.pth"
                pth_file.write_text(pth_content)

    def _install_dependencies(self):
        """Install extension dependencies into the venv, skipping already-satisfied ones."""
        if os.name == "nt":
            python_exe = self.venv_path / "Scripts" / "python.exe"
        else:
            python_exe = self.venv_path / "bin" / "python"

        if not python_exe.exists():
            raise RuntimeError(f"Python executable not found at {python_exe}")

        use_uv = self._ensure_uv()
        uv_path = shutil.which("uv") if use_uv else None

        safe_deps: list[str] = []
        for dep in self.config["dependencies"]:
            validate_dependency(dep)
            safe_deps.append(dep)

        if self.config["share_torch"] and safe_deps:
            safe_deps = self._exclude_satisfied_requirements(safe_deps, python_exe)

        if not safe_deps:
            return

        if use_uv:
            cmd_prefix = [uv_path, "pip", "install", "--python", str(python_exe)]
            cache_dir = self.venv_path.parent / ".uv_cache"
            cache_dir.mkdir(exist_ok=True)
            common_args = ["--cache-dir", str(cache_dir)]

            test_file = cache_dir / ".hardlink_test"
            test_link = self.venv_path / ".hardlink_test"
            try:
                test_file.touch()
                os.link(test_file, test_link)
                test_link.unlink()
                common_args.extend(["--link-mode", "hardlink"])
            except OSError:
                common_args.extend(["--link-mode", "copy"])
            finally:
                test_file.unlink(missing_ok=True)
        else:
            cmd_prefix = [str(python_exe), "-m", "pip", "install"]
            common_args = []

        torch_spec: Optional[str] = None
        if not self.config["share_torch"]:
            import torch
            torch_version = torch.__version__
            if torch_version.endswith("+cpu"):
                torch_version = torch_version[:-4]
            cuda_version = torch.version.cuda
            if cuda_version:
                common_args += [
                    "--extra-index-url",
                    f"https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')}",
                ]
            if "dev" in torch_version or "+" in torch_version:
                if use_uv:
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
                if cached.get("fingerprint") == fingerprint:
                    return
            except Exception:
                pass

        cmd = cmd_prefix + safe_deps + common_args

        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            assert proc.stdout is not None
            output_lines: list[str] = []
            for line in proc.stdout:
                clean = line.rstrip()
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

    def join(self):
        """Join the child process, blocking until it exits."""
        self.proc.join()
