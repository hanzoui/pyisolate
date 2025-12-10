# PyIsolate for ComfyUI Custom Nodes

**Process isolation for ComfyUI custom nodes - solve dependency conflicts without breaking your workflow.**

> 🎯 **Quick Start**: Get your custom node isolated in under 5 minutes. See [Installation](#installation) and [Converting Your Node](#converting-your-custom-node).

## What Problem Does This Solve?

ComfyUI custom nodes often require conflicting dependencies:
- Node A needs `numpy==1.24.0`
- Node B needs `numpy==2.0.0`
- Both can't coexist in the same environment

**PyIsolate solution**: Each custom node runs in its own isolated process with its own dependencies, while sharing PyTorch tensors with zero-copy performance.

## Installation

### Prerequisites
- Python 3.9+
- ComfyUI installed
- The [`uv`](https://github.com/astral-sh/uv) package manager

### Install uv (if not already installed)
```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Install PyIsolate in ComfyUI

```bash
cd ComfyUI
source .venv/bin/activate  # Windows: .venv\Scripts\activate

Clone from pollockjj's repo:
git clone https://github.com/pollockjj/pyisolate
cd pyisolate
git install .

```

### Enable Isolation in ComfyUI

Add the `--use-process-isolation` flag when launching ComfyUI:

```bash
python main.py --use-process-isolation
```

**That's it.** ComfyUI will now automatically detect and isolate any custom nodes with a `pyisolate.yaml` manifest.

---

## Converting a Custom Node

### Step 1: Create `pyisolate.yaml`

In the custom node directory, create a `pyisolate.yaml` file:

```yaml
# custom_nodes/MyAwesomeNode/pyisolate.yaml
isolated: true
share_torch: true  # Enable `zero-copy` PyTorch tensor sharing - Allows fast copy of tensors, but at a higher memory and filespace footprint

dependencies:
  - numpy==2.0.0        # Node specific numpy version
  - pillow==10.0.0      # Node specific dependencies
  - my-special-lib>=1.5
```

### Step 2: Test It

```bash
cd ComfyUI
python main.py --use-process-isolation
```

**Expected logs - Loading:**
PyIsolate and internal functions that use it use a "][" as log prefix. 
```
][ ComfyUI-IsolationTest cache miss, spawning process for metadata  # First run or cache invalidation
][ ComfyUI-PyIsolatedV3 loaded from cache                           # Subsequent runs where nodes and environment is unchanged so cache is reused
][ ComfyUI-APIsolated loaded from cache
][ ComfyUI-DepthAnythingV2 loaded from cache

][ ComfyUI-IsolationTest metadata cached
][ ComfyUI-IsolationTest ejecting after metadata extraction
```


**Expected logs - Reporting:**
```
Import times for custom nodes:
   0.0 seconds: /home/johnj/ComfyUI/custom_nodes/websocket_image_save.py
   0.0 seconds: /home/johnj/ComfyUI/custom_nodes/comfyui-florence2
   0.0 seconds: /home/johnj/ComfyUI/custom_nodes/comfyui-videohelpersuite
   0.0 seconds: /home/johnj/ComfyUI/custom_nodes/ComfyUI-GGUF
   0.0 seconds: /home/johnj/ComfyUI/custom_nodes/comfyui-kjnodes
   0.0 seconds: /home/johnj/ComfyUI/custom_nodes/ComfyUI-Manager
   0.1 seconds: /home/johnj/ComfyUI/custom_nodes/ComfyUI-Crystools
   0.3 seconds: /home/johnj/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper
   0.4 seconds: /home/johnj/ComfyUI/custom_nodes/RES4LYF


Import times for isolated custom nodes:
   0.0 seconds: /home/johnj/ComfyUI/custom_nodes/ComfyUI-DepthAnythingV2
   0.0 seconds: /home/johnj/ComfyUI/custom_nodes/ComfyUI-PyIsolatedV3
   0.0 seconds: /home/johnj/ComfyUI/custom_nodes/ComfyUI-APIsolated
   3.2 seconds: /home/johnj/ComfyUI/custom_nodes/ComfyUI-IsolationTest     #First-time cost
```


**Expected logs - during workflow usage:**
```
got prompt  # A new workflow where isolated nodes are used
][ ComfyUI-PyIsolatedV3 - just-in-time spawning of isolated custom_node
][ ComfyUI-APIsolated - just-in-time spawning of isolated custom_node
Prompt executed in 68.34 seconds

got prompt  # same workflow
Prompt executed in 61.68 seconds

got prompt  # different workflow, same two custom_nodes used
Prompt executed in 72.29 seconds

got prompt  # same 2nd workflow as above
Prompt executed in 66.17 seconds

got prompt   # new workflow, no isolated nodes used
][ ComfyUI-APIsolated isolated custom_node not in execution graph, evicting
][ ComfyUI-PyIsolatedV3 isolated custom_node not in execution graph, evicting
Prompt executed in 8.49 seconds

```

## What Works

✅ **Standard Python code execution:**
- Any standard Python code inside node functions using Comfy standard imports and each custom_node's pysiolate.yaml's dependencies
- Custom dependencies and conflicting library versions in isolated custom_nodes

✅ **Zero-copy tensor sharing (linux only):**
- PyTorch tensors pass between processes without serialization
- ~1ms overhead per RPC call
- No memory duplication

✅ **ComfyUI V3 API support (`comfy_api.latest`):**

All standard V3 API patterns work in isolated nodes:
- **I/O Types:** `IO.IMAGE`, `IO.MASK`, `IO.LATENT`, `IO.STRING`, `IO.INT`, `IO.FLOAT`, etc.
- **PromptServer calls:** `PromptServer.instance.send_sync()`, `.send()`, `.client_id`
- **Node base classes:** Inherit from `ComfyNodeABC` as usual
- **Type hints:** `InputTypeDict`, `OutputTypeDict` work normally

See [Appendix: Supported APIs](#appendix-supported-apis) for complete function lists.

✅ **ComfyUI core proxies (fully supported):**
- `model_management.py` - Device management, memory operations, interrupt handling
- `folder_paths.py` - Path resolution, model discovery, file operations
- All functions callable from isolated nodes via transparent RPC

✅ **ComfyUI standard V1 types that work across isolation:**

| Input/Output Type | Status | Notes |
|-------------------|--------|-------|
| `IMAGE` | ✅ Works | PyTorch tensor, zero-copy |
| `MASK` | ✅ Works | PyTorch tensor, zero-copy |
| `LATENT` | ✅ Works | Dict with tensor, serializes cleanly |
| `INT` | ✅ Works | Primitive type |
| `FLOAT` | ✅ Works | Primitive type |
| `STRING` | ✅ Works | Primitive type |
| `BOOLEAN` | ✅ Works | Primitive type |
| `CONDITIONING` | ✅ Works | List of tuples with tensors |
| `CONTROL_NET` | unknown | Not tested |
| `MODEL` | ⚠️ Basic | ModelPatcher object, standard inference |
| `CLIP` | ⚠️ Basic | standard CLIP decoding tested isolated |
| `VAE` | ⚠️ Basic | standard VAE decoding tested isolated |

**Key insight:** Any ComfyUI type that is fundamentally a **tensor, dict, list, or primitive** will work. Complex stateful objects like `MODEL`, `CLIP`, `VAE` cannot cross the isolation boundary (yet).

✅ **Dependency conflicts of isolated custom_nodes**
- Different numpy versions, diffusers, etc. 

---

## What Doesn't Work

❌ **PromptServer route decoration:**
```python
# This pattern does NOT work
from server import PromptServer
@PromptServer.instance.routes.get("/my_route")
def my_handler(request):
    pass
```
**Why**: Route decorators execute at module import time, before isolation is ready.  
**Workaround**: Use `route_manifest.json` (see [Advanced: Web Routes](#advanced-web-routes)).

❌ **Monkey patching ComfyUI core:**
```python
# This will NEVER work in isolation
import comfy.model_management
comfy.model_management.some_function = my_patched_version
```
**Why**: Each isolated process has its own copy of ComfyUI code. Patches don't propagate.  
**Solution**: Don't monkey patch. Use proper extension patterns instead.

---

## Live Examples

Three working isolated custom node packs are available for reference:

| Node Pack | What It Does | Isolation Benefit |
|-----------|--------------|-------------------|
| [ComfyUI-PyIsolatedV3](https://github.com/pollockjj/ComfyUI-PyIsolated) | Demo node using `deepdiff` | Shows basic isolation setup |
| [ComfyUI-APIsolated](https://github.com/pollockjj/ComfyUI-APIsolated) | API nodes (OpenAI, Gemini, etc.) | Isolated API dependencies |
| [ComfyUI-IsolationTest](https://github.com/pollockjj/ComfyUI-IsolationTest) | 70+ ComfyUI core nodes | Proves isolation doesn't break functionality |




## Performance Characteristics

### Startup Time
| Scenario | Time | Notes |
|----------|------|-------|
| **First run (cache miss)** | speed dependent environment | Creates venv, installs deps, caches metadata |
| **Subsequent runs (cache hit)** | almost instantaneous | Loads cached metadata, no spawn |
| **Process spawn on first execution** | 1-3 seconds (background) | Only when node first executes in workflow |

### Runtime Overhead
| Operation | Overhead | Impact |
|-----------|----------|--------|
| **RPC call (simple data)** | ~0.3ms | Negligible |
| **Tensor passing (share_torch)** | ~1ms | Zero-copy, minimal |
| **Large model loading** | Same as non-isolated | No overhead |

### Memory Footprint
- **Per isolated node:** ~50-300MB
- **Tensors:** Shared memory (no duplication)
- **Models:** Can be shared via ProxiedSingleton

**Bottom line:** Isolation adds ~1-2ms per node execution. For typical workflows (seconds per generation), this is <0.1% overhead.

---

## Troubleshooting

### "Cache miss, spawning process" on every startup
**Cause:** Cache invalidated (code changed, manifest changed, or Python version changed).
**Fix:** Normal behavior on first run or after updates. Subsequent runs will be fast.

### "Module not found" errors in isolated node
**Cause:** Dependency not listed in `pyisolate.yaml`.
**Fix:** Add the missing package to the `dependencies` list.

### Node works non-isolated but fails isolated
**Cause:** Likely using a pattern that doesn't work with isolation (see [What Doesn't Work](#what-doesnt-work-yet)).
**Fix:** Check logs for specific error, review the node's `__init__.py` for module-level side effects.

### "Torch already imported" warning spam
**Cause:** Isolated processes reload torch, triggering ComfyUI's warning.
**Fix:** Known issue

---

#
---

## FAQ

### Q: Can I use this on ComfyUI Cloud?
**A:** Yes, that's a primary use case. Isolation enables running diverse custom nodes without dependency conflicts in cloud environments.

### Q: Does this work with Manager?
**A:** Yes, ComfyUI Manager can install isolated nodes normally. The `pyisolate.yaml` is detected automatically.

### Q: Can I mix isolated and non-isolated nodes?
**A:** Yes, absolutely. Only nodes with `pyisolate.yaml` are isolated. Others run normally in the host process.

### Q: What about security/sandboxing?
**A:** Current focus is dependency isolation. Security sandboxing (filesystem, network restrictions) is planned for future releases.

### Q: Does this work on Windows/Mac/Linux?
**A:** Developed on linux, periodically tested on windoes.

---