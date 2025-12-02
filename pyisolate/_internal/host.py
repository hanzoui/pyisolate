import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack, contextmanager
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Generic, Optional, TypeVar

from ..config import ExtensionConfig
from ..path_helpers import serialize_host_snapshot
from ..shared import ExtensionBase
from .client import entrypoint
from .shared import AsyncRPC

logger = logging.getLogger(__name__)


def _detect_pyisolate_version() -> str:
    try:
        return importlib_metadata.version("pyisolate")
    except Exception:
        return "0.0.0"


pyisolate_version = _detect_pyisolate_version()


def normalize_extension_name(name: str) -> str:
    """
    Normalize an extension name to be safe for use in filesystem paths and shell commands.

    This function:
    - Replaces spaces and unsafe characters with underscores
    - Removes directory traversal attempts
    - Ensures the name is not empty
    - Preserves Unicode characters (for non-English names)

    Args:
        name: The original extension name

    Returns:
        A normalized, filesystem-safe version of the name

    Raises:
        ValueError: If the name is empty or only contains invalid characters
    """
    if not name:
        raise ValueError("Extension name cannot be empty")

    # Remove any directory traversal attempts or absolute path indicators
    # Replace path separators with underscores
    name = name.replace("/", "_").replace("\\", "_")

    # Remove leading dots to prevent hidden files
    while name.startswith("."):
        name = name[1:]

    # Replace consecutive dots that are part of directory traversal
    name = name.replace("..", "_")

    # Replace problematic characters with underscores
    # This includes spaces, shell metacharacters, and control characters
    # But preserves Unicode letters, numbers, and some safe punctuation
    unsafe_chars = [
        " ",  # Spaces
        "\t",  # Tabs
        "\n",  # Newlines
        "\r",  # Carriage returns
        ";",  # Command separator
        "|",  # Pipe
        "&",  # Background/and
        "$",  # Variable expansion
        "`",  # Command substitution
        "(",  # Subshell
        ")",  # Subshell
        "<",  # Redirect
        ">",  # Redirect
        '"',  # Quote
        "'",  # Quote
        "\\",  # Escape (already handled above)
        "!",  # History expansion
        "{",  # Brace expansion
        "}",  # Brace expansion
        "[",  # Glob
        "]",  # Glob
        "*",  # Glob
        "?",  # Glob
        "~",  # Home directory
        "#",  # Comment
        "%",  # Job control
        "=",  # Assignment
        ":",  # Path separator
        ",",  # Various uses
        "\0",  # Null byte
    ]

    for char in unsafe_chars:
        name = name.replace(char, "_")

    # Replace multiple consecutive underscores with a single underscore
    name = re.sub(r"_+", "_", name)

    # Remove leading and trailing underscores
    name = name.strip("_")

    # If the name is now empty (was all invalid chars), raise an error
    if not name:
        raise ValueError("Extension name contains only invalid characters")

    return name


def validate_dependency(dep: str) -> None:
    """Validate a single dependency specification."""
    if not dep:
        return

    # Special case: allow "-e" for editable installs followed by a path
    if dep == "-e":
        # This is OK, it should be followed by a path in the next argument
        return

    # Check if it looks like a command-line option (but allow -e)
    if dep.startswith("-") and not dep.startswith("-e "):
        raise ValueError(
            f"Invalid dependency '{dep}'. "
            "Dependencies cannot start with '-' as this could be a command option."
        )

    # Basic validation for common injection patterns
    # Note: We allow < and > as they're used in version specifiers
    dangerous_patterns = ["&&", "||", ";", "|", "`", "$", "\n", "\r", "\0"]
    for pattern in dangerous_patterns:
        if pattern in dep:
            raise ValueError(
                f"Invalid dependency '{dep}'. Contains potentially dangerous character: '{pattern}'"
            )


def validate_path_within_root(path: Path, root: Path) -> None:
    """Ensure a path is within the expected root directory."""
    try:
        # Resolve both paths to absolute paths
        resolved_path = path.resolve()
        resolved_root = root.resolve()

        # Check if the path is within the root
        resolved_path.relative_to(resolved_root)
    except ValueError as err:
        raise ValueError(f"Path '{path}' is not within the expected root directory '{root}'") from err


@contextmanager
def environment(**env_vars):
    """Context manager for temporarily setting environment variables"""
    original = {}

    # Save original values and set new ones
    for key, value in env_vars.items():
        original[key] = os.environ.get(key)
        os.environ[key] = str(value)

    try:
        yield
    finally:
        # Restore original values
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
        # Store original name for display purposes
        self.name = config["name"]

        # Normalize the name for filesystem operations
        self.normalized_name = normalize_extension_name(self.name)

        # Log if normalization changed the name
        if self.normalized_name != self.name:
            pass

        # Validate all dependencies
        for dep in config["dependencies"]:
            validate_dependency(dep)

        # Use Path for safer path operations with normalized name
        venv_root = Path(venv_root_path).resolve()
        self.venv_path = venv_root / self.normalized_name

        # Ensure the venv path is within the root directory
        validate_path_within_root(self.venv_path, venv_root)

        self.module_path = module_path
        self.config = config
        self.extension_type = extension_type

        if self.config["share_torch"]:
            import torch.multiprocessing

            self.mp = torch.multiprocessing
        else:
            import multiprocessing

            self.mp = multiprocessing
        
        # Initialize the isolated process
        self._initialize_process()
    
    def _filter_already_satisfied(self, requirements: list[str], python_exe: Path) -> list[str]:
        """Filter requirements list to exclude packages already satisfied in the venv.
        
        When share_torch=true, the venv has --system-site-packages, so torch and its
        dependencies from the host are visible. We check what's already installed and
        skip those packages to avoid redundant downloads.
        
        Args:
            requirements: List of package specifications from pyisolate.yaml
            python_exe: Path to venv python executable
            
        Returns:
            Filtered list of requirements that still need installation
        """
        import subprocess
        from packaging.requirements import Requirement
        from packaging.specifiers import SpecifierSet
        
        # Get all installed packages in venv (including system-site-packages)
        result = subprocess.run(
            [str(python_exe), "-m", "pip", "list", "--format", "json"],
            capture_output=True,
            text=True,
            check=True
        )
        installed = {pkg["name"].lower(): pkg["version"] for pkg in json.loads(result.stdout)}
        
        # Torch-related packages that should never be installed when share_torch=true
        TORCH_ECOSYSTEM = {
            'torch', 'torchvision', 'torchaudio', 'torchtext', 'triton',
            'nvidia-cuda-runtime-cu12', 'nvidia-cuda-nvrtc-cu12', 'nvidia-cudnn-cu12',
            'nvidia-cublas-cu12', 'nvidia-cufft-cu12', 'nvidia-curand-cu12',
            'nvidia-cusolver-cu12', 'nvidia-cusparse-cu12', 'nvidia-nccl-cu12',
            'nvidia-nvtx-cu12', 'nvidia-nvjitlink-cu12', 'nvidia-cuda-cupti-cu12',
            'nvidia-cusparselt-cu12', 'nvidia-nvshmem-cu12', 'nvidia-cufile-cu12'
        }
        
        filtered = []
        for req_str in requirements:
            # Handle editable installs (-e path) - pass through unchanged
            req_str_stripped = req_str.strip()
            if req_str_stripped.startswith('-e ') or req_str_stripped == '-e':
                # Editable install - always include, skip parsing
                filtered.append(req_str)
                continue
            
            # Handle bare paths (editable without -e flag or local installs)
            if req_str_stripped.startswith('/') or req_str_stripped.startswith('./'):
                filtered.append(req_str)
                continue
            
            try:
                req = Requirement(req_str)
                pkg_name_lower = req.name.lower()
                
                # Skip torch ecosystem packages when share_torch=true
                if self.config["share_torch"] and pkg_name_lower in TORCH_ECOSYSTEM:
                    continue
                
                # Check if package is installed and version satisfies requirement
                if pkg_name_lower in installed:
                    installed_version = installed[pkg_name_lower]
                    if not req.specifier or installed_version in req.specifier:
                        continue
                    else:
                        pass
                
                filtered.append(req_str)
                
            except Exception as e:
                # If we can't parse, include it to be safe
                logger.warning(
                    "📚 [PyIsolate][Deps] Could not parse requirement '%s': %s (including anyway)",
                    req_str,
                    e
                )
                filtered.append(req_str)
        
        return filtered
    
    def _initialize_process(self) -> None:
        """Initialize the isolated process with queues and RPC."""
        # Use get_context instead of set_start_method for better isolation
        # and to avoid "context already set" errors
        try:
            self.ctx = self.mp.get_context("spawn")
        except ValueError as e:
            raise RuntimeError(
                f"Failed to get 'spawn' context for pyisolate: {e}. "
                "Pyisolate requires the 'spawn' start method to work correctly."
            ) from e
        
        # Use Manager-based queues for Windows cross-venv compatibility
        # Direct Queue passing fails on Windows with set_executable() due to
        # handle inheritance issues (WinError 5: Access Denied on semaphores)
        #
        # CRITICAL: Set PYISOLATE_CHILD=1 BEFORE creating Manager!
        # On Windows, Manager.start() spawns a new process that re-runs main.py.
        # Without this env var, ComfyUI's IS_PRIMARY_PROCESS check fails and it
        # tries to start the full server, causing import errors.
        import multiprocessing as std_mp
        os.environ["PYISOLATE_CHILD"] = "1"
        try:
            std_ctx = std_mp.get_context("spawn")
            self.manager = std_ctx.Manager()
        except Exception as e:
            raise RuntimeError(
                f"Failed to start multiprocessing Manager for {self.name}: {e}"
            ) from e
        
        self.to_extension = self.manager.Queue()
        self.from_extension = self.manager.Queue()
        self.extension_proxy = None
        try:
            self.proc = self.__launch()
        except Exception as exc:
            logger.error("📚 [PyIsolate][Extension] Launch failed for %s: %s", self.name, exc)
            raise
        self.rpc = AsyncRPC(recv_queue=self.from_extension, send_queue=self.to_extension)
        for api in self.config["apis"]:
            api()._register(self.rpc)
        self.rpc.run()

    def get_proxy(self) -> T:
        if self.extension_proxy is None:
            self.extension_proxy = self.rpc.create_caller(self.extension_type, "extension")

        return self.extension_proxy

    def stop(self) -> None:
        """Stop the extension process and clean up resources."""
        errors: list[str] = []

        if hasattr(self, "proc") and self.proc.is_alive():
            try:
                self.proc.terminate()
                self.proc.join(timeout=5.0)
                if self.proc.is_alive():
                    logger.warning(
                        "📚 [PyIsolate][Extension] Force killing hung extension %s", self.name
                    )
                    self.proc.kill()
                    self.proc.join()
            except Exception as exc:  # pragma: no cover - depends on multiprocessing edge cases
                detail = f"Failed to terminate process for {self.name}: {exc}"
                logger.error(detail)
                errors.append(detail)

        for attr_name in ("to_extension", "from_extension"):
            queue = getattr(self, attr_name, None)
            if queue is None:
                continue
            try:
                queue.close()
            except Exception as exc:  # pragma: no cover - depends on multiprocessing edge cases
                detail = f"Failed to close {attr_name} queue for {self.name}: {exc}"
                logger.error(detail)
                errors.append(detail)

        # Shutdown the Manager (required for Windows Manager-based queues)
        if hasattr(self, "manager") and self.manager is not None:
            try:
                self.manager.shutdown()
            except Exception as exc:  # pragma: no cover
                detail = f"Failed to shutdown manager for {self.name}: {exc}"
                logger.error(detail)
                errors.append(detail)

        if errors:
            raise RuntimeError("; ".join(errors))
    def __launch(self):
        """
        Launch the extension in a separate process.
        """
        # Create the virtual environment for the extension
        self._create_extension_venv()

        # Install dependencies in the virtual environment
        self._install_dependencies()

        # Set the Python executable from the virtual environment
        if os.name == "nt":
            executable = str(self.venv_path / "Scripts" / "python.exe")
        else:
            executable = str(self.venv_path / "bin" / "python")
        
        # Capture host sys.path snapshot for child reconstruction
        snapshot_file = Path(tempfile.gettempdir()) / f"pyisolate_snapshot_{self.name}.json"
        snapshot = serialize_host_snapshot(output_path=str(snapshot_file))
        
        self.mp.set_executable(executable)
        with ExitStack() as stack:
            stack.enter_context(
                environment(
                    PYISOLATE_CHILD="1",
                    PYISOLATE_EXTENSION=self.name,
                    PYISOLATE_MODULE_PATH=self.module_path,
                    PYISOLATE_HOST_SNAPSHOT=str(snapshot_file),
                )
            )
            if os.name == "nt":
                stack.enter_context(
                    environment(
                        VIRTUAL_ENV=str(self.venv_path),
                    )
                )
            # Use context-based Process for consistent spawn behavior
            proc = self.ctx.Process(
                target=entrypoint,
                args=(
                    self.module_path,
                    self.extension_type,
                    self.config,
                    self.to_extension,
                    self.from_extension,
                ),
            )
            proc.start()

        return proc

    def _ensure_uv(self) -> bool:
        """
        Ensure 'uv' is available. If not found, attempt to install it via pip.
        Returns True if uv is available (either found or installed), False otherwise.
        """
        if shutil.which("uv"):
            return True

        logger.warning("📚 [PyIsolate][Setup] ⚠️ 'uv' not found. Attempting to install via pip...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "uv"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            if shutil.which("uv"):
                logger.info("📚 [PyIsolate][Setup] ✅ 'uv' installed successfully.")
                return True
        except Exception as e:
            logger.warning(f"📚 [PyIsolate][Setup] ⚠️ Failed to install 'uv': {e}")
        
        logger.warning("📚 [PyIsolate][Setup] ⚠️ Falling back to standard pip/venv (slower).")
        return False

    def _create_extension_venv(self):
        """
        Create a virtual environment for the extension if it doesn't exist.
        """
        # Ensure parent directory exists
        self.venv_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.venv_path.exists():
            logger.info(
                "📚 [PyIsolate][Venv] Creating virtual environment name=%s path=%s share_torch=%s",
                self.name,
                self.venv_path,
                self.config["share_torch"],
            )

            # Create venv using host Python
            # CRITICAL ARCHITECTURE CHANGE:
            # We do NOT use --system-site-packages because it points to /usr/bin (system python),
            # causing us to inherit broken system packages (like matplotlib) instead of our parent venv.
            # Instead, we create a clean venv and inject the parent venv via a .pth file.
            cmd = [sys.executable, "-m", "venv", str(self.venv_path)]
            
            try:
                subprocess.check_call(cmd)  # noqa: S603
                
                # Configure inheritance via .pth file injection
                if self.config["share_torch"]:
                    import site
                    
                    # 1. Identify child site-packages directory
                    if os.name == "nt":
                        child_site = self.venv_path / "Lib" / "site-packages"
                    else:
                        # POSIX: lib/pythonX.Y/site-packages
                        version_info = sys.version_info
                        child_site = self.venv_path / "lib" / f"python{version_info.major}.{version_info.minor}" / "site-packages"
                    
                    if not child_site.exists():
                        # Fallback discovery if path structure differs
                        logger.warning("📚 [PyIsolate][Venv] Standard site-packages path not found at %s, probing...", child_site)
                        # This is rare for standard venv, but safety first
                        found = list(self.venv_path.glob("**/site-packages"))
                        if found:
                            child_site = found[0]
                        else:
                            raise RuntimeError(f"Could not locate site-packages in new venv: {self.venv_path}")

                    # 2. Identify parent site-packages (from current host process)
                    # We want the primary site-packages of the host venv
                    parent_sites = site.getsitepackages()
                    # Filter for the one inside our .venv_golden (host)
                    # Usually the first one, but let's be precise
                    host_prefix = sys.prefix
                    valid_parents = [p for p in parent_sites if p.startswith(host_prefix)]
                    
                    if not valid_parents:
                        # Fallback: use the first one or sys.path calculation
                        logger.warning("📚 [PyIsolate][Venv] Could not strictly identify parent site-packages in %s, using sys.path heuristic", parent_sites)
                        valid_parents = [p for p in sys.path if "site-packages" in p and p.startswith(host_prefix)]
                    
                    if not valid_parents:
                         raise RuntimeError("Could not determine parent site-packages path to inherit")

                    parent_site = valid_parents[0]
                    
                    # 3. Write the .pth file
                    # Using site.addsitedir() ensures .pth files in the parent are processed too
                    pth_content = f"import site; site.addsitedir(r'{parent_site}')\n"
                    pth_file = child_site / "_pyisolate_parent.pth"
                    pth_file.write_text(pth_content)
                    
                    logger.info(
                        "📚 [PyIsolate][Venv] ✅ Configured inheritance: Child -> %s",
                        parent_site
                    )
                
                logger.info("📚 [PyIsolate][Venv] ✅ Created venv at %s", self.venv_path)
            except subprocess.CalledProcessError as e:
                logger.error("📚 [PyIsolate][Venv] ❌ Failed to create venv: %s", e)
                raise

    # TODO(Optimization): Only do this when we update a extension to reduce startup time?
    def _install_dependencies(self):
        """
        Install dependencies in the extension's virtual environment.
        """
        if os.name == "nt":
            python_executable = self.venv_path / "Scripts" / "python.exe"
        else:
            python_executable = self.venv_path / "bin" / "python"

        # Ensure the Python executable exists
        if not python_executable.exists():
            raise RuntimeError(f"Python executable not found at {python_executable}")

        # Determine if we can use uv
        use_uv = self._ensure_uv()
        uv_path = shutil.which("uv") if use_uv else None

        # Install extension dependencies from config
        safe_dependencies: list[str] = []
        for dep in self.config["dependencies"]:
            validate_dependency(dep)
            safe_dependencies.append(dep)

        # Filter requirements to exclude packages already satisfied by host environment
        if self.config["share_torch"] and safe_dependencies:
            safe_dependencies = self._filter_already_satisfied(safe_dependencies, python_executable)
        
        install_required = bool(safe_dependencies) or self.config["share_torch"]
        if not install_required:
            return

        # Prepare command prefix and args
        if use_uv:
            cmd_prefix = [uv_path, "pip", "install", "--python", str(python_executable)]
            common_args = []
            # Set up a local cache directory next to venvs to ensure same filesystem
            cache_dir = self.venv_path.parent / ".uv_cache"
            cache_dir.mkdir(exist_ok=True)
            common_args.extend(["--cache-dir", str(cache_dir)])
            
            # Use hardlinks for near-instant installation from cache
            # Note: Falls back to copy automatically if hardlinks fail (cross-filesystem)
            try:
                # Test if hardlinks work between cache and venv
                test_file = cache_dir / ".hardlink_test"
                test_link = self.venv_path / ".hardlink_test"
                test_file.touch()
                try:
                    os.link(test_file, test_link)
                    test_link.unlink()
                    common_args.extend(["--link-mode", "hardlink"])
                except OSError:
                    # Cross-filesystem, use copy mode
                    common_args.extend(["--link-mode", "copy"])
                finally:
                    test_file.unlink(missing_ok=True)
            except Exception as e:
                # Fallback: don't specify link-mode, let uv decide
                pass
        else:
            cmd_prefix = [str(python_executable), "-m", "pip", "install"]
            common_args = []

        torch_spec: Optional[str] = None
        torch_constraint_file: Optional[Path] = None
        
        if self.config["share_torch"]:
            import torch
            torch_version = torch.__version__
            if torch_version.endswith("+cpu"):
                torch_version = torch_version[:-4]         
        else:
            import torch

            torch_version = torch.__version__
            if torch_version.endswith("+cpu"):
                torch_version = torch_version[:-4]
            cuda_version = torch.version.cuda  # type: ignore
            if cuda_version:
                common_args += [
                    "--extra-index-url",
                    f"https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')}",
                ]

            if "dev" in torch_version or "+" in torch_version:
                if use_uv:
                    common_args.append("--index-strategy")
                    common_args.append("unsafe-best-match")
                # pip handles this differently or ignores it, usually fine for simple cases

            torch_spec = f"torch=={torch_version}"
            safe_dependencies.insert(0, torch_spec)
            logger.info(
                "📚 [PyIsolate][Deps] Installing torch %s in fully isolated venv for %s",
                torch_spec,
                self.name,
            )

        descriptor = {
            "dependencies": safe_dependencies,
            "share_torch": self.config["share_torch"],
            "torch_spec": torch_spec,
            "pyisolate": pyisolate_version,
            "python": sys.version,
        }
        fingerprint = hashlib.sha256(
            json.dumps(descriptor, sort_keys=True).encode("utf-8")
        ).hexdigest()
        lock_path = self.venv_path / ".pyisolate_deps.json"
        if lock_path.exists():
            try:
                cached = json.loads(lock_path.read_text(encoding="utf-8"))
            except Exception:
                cached = {}
            if cached.get("fingerprint") == fingerprint:
                return

        cmd = cmd_prefix + safe_dependencies + common_args

        try:
            with subprocess.Popen(  # noqa: S603
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
                    if "pyisolate==" in clean or "pyisolate @" in clean:
                        continue
                    output_lines.append(clean)
                return_code = proc.wait()
            if return_code != 0:
                detail = "\n".join(output_lines) or "(no output)"
                msg = (
                    f"📚 [PyIsolate][Deps] Install failed for {self.name} "
                    f"returncode={return_code} command={' '.join(cmd)} output={detail}"
                )
                logger.error(msg)
                raise RuntimeError(msg)
            # Summary log for user visibility
            skipped_count = len(self.config["dependencies"]) - len(safe_dependencies)
            if self.config["share_torch"] and torch_spec:
                 # torch_spec was added to safe_dependencies, so adjust count
                 skipped_count += 1
            
            logger.info(
                "📚 [PyIsolate][Deps] ✅ Installed dependencies for %s (Filtered %d cached)",
                self.name,
                max(0, skipped_count)
            )

            lock_path.write_text(
                json.dumps({"fingerprint": fingerprint, "descriptor": descriptor}, indent=2),
                encoding="utf-8",
            )
        except OSError as e:  # Includes FileNotFoundError, BrokenPipeError, etc.
            msg = (
                f"📚 [PyIsolate][Deps] Install failed for {self.name} due to OS error "
                f"{e.__class__.__name__}: {e}"
            )
            logger.error(msg)
            raise RuntimeError(msg) from e

    def join(self):
        """
        Wait for the extension process to finish.
        """
        self.proc.join()
