# Platform Compatibility Matrix

This document describes pyisolate's compatibility across different platforms, operating systems, and configurations.

## Operating System Support

| OS | Sandbox | Tensor IPC | Notes |
|----|---------|------------|-------|
| Linux (glibc) | ✅ Full | ✅ Full | Primary supported platform |
| Linux (musl/Alpine) | ✅ Full | ✅ Full | Requires bubblewrap |
| Ubuntu 22.04+ | ⚠️ Degraded | ✅ Full | AppArmor restricts user namespaces |
| Ubuntu 20.04 | ✅ Full | ✅ Full | |
| Debian 11+ | ✅ Full | ✅ Full | |
| RHEL/CentOS 8+ | ✅ Full | ✅ Full | |
| Fedora | ✅ Full | ✅ Full | SELinux may need configuration |
| macOS | ❌ None | ⚠️ Limited | No namespace support |
| Windows | ❌ None | ❌ None | Use WSL2 |
| WSL2 | ⚠️ Varies | ✅ Full | Depends on kernel version |

### Legend
- ✅ Full: All features work as designed
- ⚠️ Degraded/Limited: Some features unavailable or restricted
- ❌ None: Feature not supported

## Sandbox Mode Details

### Linux with Full Support

```
RestrictionModel.NONE
```

Full sandbox capabilities available:
- User namespace isolation
- PID namespace isolation
- Filesystem mount namespace
- Network isolation (optional)

### Ubuntu 22.04+ (AppArmor Restricted)

```
RestrictionModel.APPARMOR
```

Ubuntu's AppArmor profile restricts bubblewrap's `--unshare-user` flag. PyIsolate automatically detects this and runs in degraded mode:

```python
from pyisolate._internal.sandbox_detect import detect_restriction_model, RestrictionModel

model = detect_restriction_model()
if model == RestrictionModel.APPARMOR:
    print("Running in degraded sandbox mode")
```

**Impact**:
- No user namespace isolation
- Filesystem isolation still works
- Process runs as current user

**Workaround** (requires root):
```bash
# Disable AppArmor for bwrap (not recommended for production)
sudo ln -s /etc/apparmor.d/bwrap /etc/apparmor.d/disable/
sudo apparmor_parser -R /etc/apparmor.d/bwrap
```

### macOS

macOS does not support Linux namespaces. PyIsolate runs without sandbox:

```
SandboxMode.DISABLED
```

Extensions run in the same environment as the host process with no isolation.

## Tensor IPC Compatibility

### CPU Tensor Transfer

| Platform | Method | Performance |
|----------|--------|-------------|
| Linux | POSIX shared memory (/dev/shm) | ✅ Zero-copy |
| Linux (no /dev/shm) | File-based fallback | ⚠️ Copy required |
| macOS | File-based | ⚠️ Copy required |
| Windows (native) | Not supported | ❌ |
| WSL2 | POSIX shared memory | ✅ Zero-copy |

### CUDA Tensor Transfer

| Platform | Method | Performance |
|----------|--------|-------------|
| Linux + NVIDIA GPU | CUDA IPC handles | ✅ Zero-copy |
| macOS + Apple Silicon | Not supported | ❌ |
| Windows (native) | Not supported | ❌ |
| WSL2 + NVIDIA GPU | CUDA IPC | ✅ Zero-copy |

**Requirements for CUDA IPC**:
1. `PYISOLATE_ENABLE_CUDA_IPC=1` environment variable
2. Same CUDA version in host and extension
3. Same GPU device visible to both processes
4. Sufficient GPU memory

## Python Version Compatibility

| Python Version | Status | Notes |
|---------------|--------|-------|
| 3.12 | ✅ Supported | Primary development version |
| 3.11 | ✅ Supported | |
| 3.10 | ✅ Supported | |
| 3.9 | ⚠️ Limited | May lack some type annotations |
| 3.8 | ❌ Not supported | Missing required features |

## PyTorch Version Compatibility

| PyTorch Version | Status | Notes |
|----------------|--------|-------|
| 2.x | ✅ Supported | Recommended |
| 1.13 | ⚠️ Limited | IPC API differences |
| 1.12 | ⚠️ Limited | Some tensor operations differ |
| < 1.12 | ❌ Not supported | |

## Container Support

### Docker

| Configuration | Sandbox | Notes |
|--------------|---------|-------|
| Default | ❌ None | Lacks required capabilities |
| `--privileged` | ✅ Full | Full capabilities, less secure |
| `--cap-add SYS_ADMIN` | ✅ Full | Minimal required capability |
| `--security-opt apparmor=unconfined` | ⚠️ Varies | Depends on base image |

**Recommended Docker configuration**:
```dockerfile
# In docker run:
docker run --cap-add SYS_ADMIN --security-opt seccomp=unconfined ...
```

### Kubernetes

For Kubernetes pods:
```yaml
securityContext:
  capabilities:
    add:
      - SYS_ADMIN
  seccompProfile:
    type: Unconfined
```

**Note**: Many Kubernetes clusters restrict these capabilities for security reasons.

## Hardware Requirements

### Minimum
- 2GB RAM
- 100MB /dev/shm space
- Single CPU core

### Recommended for GPU Workloads
- 8GB+ RAM
- 1GB+ /dev/shm space
- NVIDIA GPU with 4GB+ VRAM
- NVIDIA driver 470+
- CUDA 11.x or 12.x

## Feature Detection

Use these utilities to check platform capabilities at runtime:

```python
from pyisolate._internal.sandbox_detect import (
    detect_restriction_model,
    RestrictionModel,
)

# Check sandbox capability
model = detect_restriction_model()
print(f"Restriction model: {model}")

if model == RestrictionModel.NONE:
    print("Full sandbox support available")
elif model == RestrictionModel.APPARMOR:
    print("Running with AppArmor restrictions")
elif model == RestrictionModel.SYSCTL:
    print("User namespaces disabled via sysctl")

# Check /dev/shm
from pyisolate._internal.tensor_serializer import _check_shm_availability
if _check_shm_availability():
    print("/dev/shm available for tensor IPC")

# Check CUDA IPC
import os
if os.environ.get("PYISOLATE_ENABLE_CUDA_IPC") == "1":
    try:
        import torch
        if torch.cuda.is_available():
            print("CUDA IPC enabled")
    except ImportError:
        print("PyTorch not available")
```

## Troubleshooting Platform Issues

### Linux: Enable User Namespaces

```bash
# Check current setting
cat /proc/sys/kernel/unprivileged_userns_clone

# Enable (requires root)
sudo sysctl -w kernel.unprivileged_userns_clone=1

# Make persistent
echo 'kernel.unprivileged_userns_clone=1' | sudo tee /etc/sysctl.d/99-userns.conf
```

### Ubuntu: Check AppArmor Status

```bash
# Check if bwrap is restricted
aa-status | grep bwrap

# Check current AppArmor mode
cat /sys/module/apparmor/parameters/enabled
```

### Docker: Verify Capabilities

```bash
# Inside container
capsh --print | grep cap_sys_admin
```

### WSL2: Check Kernel Version

```bash
uname -r
# Should be 5.10+ for best compatibility
```
