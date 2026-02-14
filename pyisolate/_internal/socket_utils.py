"""Platform-agnostic socket path utilities for PyIsolate IPC."""

import os
import socket
import tempfile
from pathlib import Path

__all__ = ["get_ipc_socket_dir", "ensure_ipc_socket_dir", "has_af_unix"]


def has_af_unix() -> bool:
    """Check if AF_UNIX is available on this platform."""
    return hasattr(socket, "AF_UNIX")


def get_ipc_socket_dir() -> Path:
    """Return platform-appropriate directory for IPC sockets.

    Linux: /run/user/{uid}/pyisolate (XDG_RUNTIME_DIR pattern)
    Windows: %TEMP%/pyisolate (AF_UNIX supported in Python 3.10+ on Windows 10+)
    """
    if os.name == "nt":
        # Windows: Use temp directory for AF_UNIX sockets
        return Path(tempfile.gettempdir()) / "pyisolate"
    else:
        # Linux/Unix: Use XDG_RUNTIME_DIR or fallback
        uid = os.getuid()  # type: ignore[attr-defined]  # Only called on Unix
        run_dir = Path(f"/run/user/{uid}/pyisolate")
        if not run_dir.parent.exists():
            run_dir = Path(f"/tmp/pyisolate-{uid}")
        return run_dir


def ensure_ipc_socket_dir() -> Path:
    """Create and return the IPC socket directory with appropriate permissions."""
    socket_dir = get_ipc_socket_dir()
    if os.name == "nt":
        # Windows: mkdir without mode (permissions handled by OS)
        socket_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Linux/Unix: Secure permissions
        socket_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    return socket_dir
