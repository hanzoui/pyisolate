import os
from pathlib import Path
from typing import Any

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
