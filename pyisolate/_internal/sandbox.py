import os
import sys
from pathlib import Path
from typing import Any, Optional

from .sandbox_detect import RestrictionModel

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


def build_bwrap_command(
    python_exe: str,
    module_path: str,
    venv_path: str,
    uds_address: str,
    allow_gpu: bool = False,
    sandbox_config: Optional[dict[str, Any]] = None,
    restriction_model: RestrictionModel = RestrictionModel.NONE,
    env_overrides: Optional[dict[str, str]] = None,
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
        restriction_model: Detected system restriction model

    Returns:
        Command list suitable for subprocess.Popen
    """
    if sandbox_config is None:
        sandbox_config = {}

    cmd = ["bwrap"]

    # Namespace Isolation Logic
    # -------------------------
    # We attempt full user namespace isolation (--unshare-user) if possible.
    # This allows us to map UIDs and isolate the process fully.
    # However, modern distros often restrict this (AppArmor, sysctl).

    if restriction_model == RestrictionModel.NONE:
        # Full isolation available
        cmd.extend(["--unshare-user", "--unshare-pid"])
        # We do NOT unshare-ipc because CUDA shared memory (legacy) and
        # Python SharedMemory lease require /dev/shm access in the host namespace
        # (or a shared namespace). Since we bind /dev/shm, we keep IPC shared.
    else:
        # Run in degraded mode (no user/pid namespace)
        # We still get filesystem isolation via mount namespace (bwrap default).
        pass

    # New session (detach from terminal)
    cmd.append("--new-session")

    # Ensure child dies when parent dies (prevent zombie processes)
    cmd.append("--die-with-parent")

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
        cmd.extend(["--ro-bind", "/sys", "/sys"])
        dev_path = Path("/dev")
        for pattern in GPU_PASSTHROUGH_PATTERNS:
            for dev in dev_path.glob(pattern):
                if dev.exists():
                    cmd.extend(["--dev-bind", str(dev), str(dev)])
        # CUDA IPC requires shared memory
        # SECURITY: /dev/shm is shared. This is a known side-channel risk
        # but unavoidable for zero-copy tensor transfer. Document this trade-off.
        # MOVED: /dev/shm binding is now global (see below) because CPU tensors need it too.

        # CUDA library and runtime paths (read-only)
        # /usr/local/cuda is covered by /usr bind, so we skip it to avoid symlink/mount issues
        cuda_paths = {
            "/opt/cuda",              # Alternative CUDA location
            "/run/nvidia-persistenced",  # Persistence daemon
        }

        # Add CUDA_HOME if set and not in /usr (redundant)
        cuda_home = os.environ.get("CUDA_HOME")
        if cuda_home:
            cuda_paths.add(cuda_home)

        for cuda_path in cuda_paths:
            if os.path.exists(cuda_path):
                 # Skip if already covered by /usr bind
                if cuda_path.startswith("/usr/") and not cuda_path.startswith("/usr/local/"):
                     # Actually /usr/local is in /usr.
                     # Safe heuristic: if it starts with /usr, we assume covered.
                     continue
                cmd.extend(["--ro-bind", cuda_path, cuda_path])

    # Network: ISOLATED by default (as per tests)
    cmd.append("--unshare-net")
    # If loopback is needed, we might need --share-net or explicit loopback setup.
    # But tests enforce --unshare-net.

    # Additional paths from config (user-specified)
    for path in sandbox_config.get("writable_paths", []):
        if os.path.exists(path):
            cmd.extend(["--bind", path, path])

    ro_paths = sandbox_config.get("readonly_paths", [])
    if isinstance(ro_paths, list):
        for path in ro_paths:
            if os.path.exists(path):
                cmd.extend(["--ro-bind", path, path])
    elif isinstance(ro_paths, dict):
        for src, dst in ro_paths.items():
            if os.path.exists(src):
                cmd.extend(["--ro-bind", src, dst])


    # ---------------------------------------------------------------------------
    # CRITICAL: BINDING HOST DEPENDENCIES (Refactor Branch Logic)
    # ---------------------------------------------------------------------------

    # 1. Host venv site-packages: READ-ONLY (for share_torch inheritance via .pth file)
    # The child venv has a .pth file pointing to host site-packages for torch sharing
    # We find where 'torch' is likely installed (host site-packages)
    host_site_packages = Path(sys.executable).parent.parent / "lib"
    for sp in host_site_packages.glob("python*/site-packages"):
        if sp.exists():
            cmd.extend(["--ro-bind", str(sp), str(sp)])
            break

    # 2. PyIsolate package path: READ-ONLY (needed for sandbox_client/uds_client)
    import pyisolate as pyisolate_pkg
    pyisolate_path = Path(pyisolate_pkg.__file__).parent.parent.resolve()
    cmd.extend(["--ro-bind", str(pyisolate_path), str(pyisolate_path)])

    # 3. ComfyUI package path: READ-ONLY (needed for comfy.isolation.adapter)
    try:
        import comfy  # type: ignore[import]
        if hasattr(comfy, "__file__") and comfy.__file__:
            comfy_path = Path(comfy.__file__).parent.parent.resolve()
        elif hasattr(comfy, "__path__"):
             # Namespace package support
            comfy_path = Path(list(comfy.__path__)[0]).parent.resolve()
        else:
             comfy_path = None

        if comfy_path:
            cmd.extend(["--ro-bind", str(comfy_path), str(comfy_path)])
    except Exception:
        pass

    # Shared Memory (REQUIRED for zero-copy tensors via SharedMemory Lease)
    if Path("/dev/shm").exists():
        cmd.extend(["--bind", "/dev/shm", "/dev/shm"])

    # UDS socket directory must be accessible
    uds_dir = os.path.dirname(uds_address)
    if uds_dir:
        # Create parent directories for UDS mount point to ensure they exist in tmpfs structure
        parts = Path(uds_dir).parts
        current = Path("/")
        for part in parts[1:]:
            current = current / part
            cmd.extend(["--dir", str(current)])

    if uds_dir and os.path.exists(uds_dir):
        cmd.extend(["--bind", uds_dir, uds_dir])

    # Environment variables
    cmd.extend(["--setenv", "PYISOLATE_UDS_ADDRESS", uds_address])
    cmd.extend(["--setenv", "PYISOLATE_CHILD", "1"])

    # 4. Set PYTHONPATH to include pyisolate package
    # This ensures the child can find 'pyisolate' even if not installed in its venv
    pyisolate_parent = str(pyisolate_path)
    # Start with our explicitly bound package
    new_pythonpath_parts = [pyisolate_parent]

    # Check existing PYTHONPATH
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    if existing_pythonpath:
        new_pythonpath_parts.append(existing_pythonpath)

    cmd.extend(["--setenv", "PYTHONPATH", ":".join(new_pythonpath_parts)])

    # Inherit select environment variables
    # Standard environment
    for env_var in ["PATH", "HOME", "LANG", "LC_ALL"]:
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
        "PYISOLATE_ENABLE_CUDA_IPC",
    ]
    for env_var in cuda_env_vars:
        if env_var in os.environ:
            cmd.extend(["--setenv", env_var, os.environ[env_var]])

    # Coverage / Profiling forwarding
    for key, val in os.environ.items():
        if key.startswith("COV_") or key.startswith("COVERAGE_"):
            cmd.extend(["--setenv", key, val])

    # Env overrides from config
    if env_overrides:
        for key, val in env_overrides.items():
            cmd.extend(["--setenv", key, val])

    # Command to run (Corrected to uds_client for main branch architecture)
    cmd.extend([python_exe, "-m", "pyisolate._internal.uds_client"])

    return cmd
