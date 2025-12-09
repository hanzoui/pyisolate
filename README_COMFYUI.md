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

# Install pyisolate
pip install pyisolate

# Or install from source for latest features
pip install git+https://github.com/Comfy-Org/pyisolate.git
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
share_torch: true  # Enable zero-copy PyTorch tensor sharing

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

**Expected logs:**
```
][ MyAwesomeNode loaded from cache
][ MyAwesomeNode - just-in-time spawning of isolated custom_node
```

### Step 3: Verify Isolation Works

Run a workflow that uses the node. Check that:
- ✅ Node executes successfully
- ✅ No dependency conflicts appear
- ✅ Performance is acceptable (first run spawns process, subsequent runs reuse it)

---

## What Works

✅ **Standard Python code execution:**
- Any standard Python code inside node functions using Comfy standard imports and each custom_node's pysiolate.yaml's dependencies
- Custom dependencies and conflicting library versions in isolated custom_nodes

✅ **Zero-copy tensor sharing:**
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
| `CONTROL_NET` | ⚠️ Partial | Depends on structure |
| `MODEL` | ❌ Doesn't work | ModelPatcher object, not serializable |
| `CLIP` | ❌ Doesn't work | Complex object with state |
| `VAE` | ❌ Doesn't work | Complex object with state |

**Key insight:** Any ComfyUI type that is fundamentally a **tensor, dict, list, or primitive** will work. Complex stateful objects like `MODEL`, `CLIP`, `VAE` cannot cross the isolation boundary (yet).

✅ **Dependency conflicts of isolated custom_nodes**
- Different numpy versions, diffusers, etc. 

---

## What Doesn't Work

❌ **Complex ComfyUI objects:**
- `ModelPatcher` objects - Cannot be serialized across process boundary
- `ModelSampler` objects - Tied to host process state
- Full `CLIP` objects - Use proxied versions instead
- Full `VAE` objects - Use proxied versions instead

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
| **First run (cache miss)** | ~200-500ms per node | Creates venv, installs deps, caches metadata |
| **Subsequent runs (cache hit)** | ~5-20ms per node | Loads cached metadata, no spawn |
| **Process spawn on first execution** | ~200-500ms | Only when node first executes in workflow |

### Runtime Overhead
| Operation | Overhead | Impact |
|-----------|----------|--------|
| **RPC call (simple data)** | ~0.3ms | Negligible |
| **Tensor passing (share_torch)** | ~1ms | Zero-copy, minimal |
| **Large model loading** | Same as non-isolated | No overhead |

### Memory Footprint
- **Per isolated node:** ~50-100MB (process overhead)
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
**Fix:** This is cosmetic and harmless. The warning is suppressed in PyIsolate >=0.2.0.

---

## Advanced: Web Routes

If a node registers web routes, you need a `route_manifest.json`:

```json
{
  "routes": [
    {
      "method": "GET",
      "path": "/my_node/status",
      "handler": "get_status"
    },
    {
      "method": "POST",
      "path": "/my_node/process",
      "handler": "process_data"
    }
  ]
}
```

Then define handlers as async methods in your extension class:

```python
# __init__.py
class MyNodeExtension(ExtensionBase):
    async def get_status(self, request):
        return {"status": "ok"}
    
    async def process_data(self, request):
        data = await request.json()
        result = await self.extension.process(data)
        return {"result": result}
```

Routes are automatically injected by PyIsolate's route injection system.

---

## Advanced: Shared Services (ProxiedSingleton)

Share state across all isolated nodes:

```python
# In ComfyUI core or shared module
from pyisolate import ProxiedSingleton

class ModelCache(ProxiedSingleton):
    def __init__(self):
        self.models = {}
    
    async def get_model(self, name):
        if name not in self.models:
            self.models[name] = load_expensive_model(name)
        return self.models[name]
```

```python
# In isolated node
class MyNode:
    def process(self, model_name):
        cache = ModelCache()  # Returns proxy to host's singleton
        model = await cache.get_model(model_name)
        return model.predict(...)
```

This allows expensive resources (models, database connections) to be shared across all isolated nodes without duplication.

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
**A:** Yes, tested on all three platforms.

---

## Getting Help

- **GitHub Issues:** https://github.com/Comfy-Org/pyisolate/issues
- **Example Nodes:** See [Live Examples](#live-examples) section
- **Technical Docs:** https://comfy-org.github.io/pyisolate/

---

## For Tomorrow's Demo

### What to Show
1. ✅ **Process isolation working**: Run workflow with isolated nodes, show logs
2. ✅ **Converting a node pack**: Live demo of adding `pyisolate.yaml`
3. ✅ **What "just works"**: Show ComfyUI-IsolationTest (70+ nodes working)
4. ✅ **What doesn't work**: Explain web routes limitation, show route manifest solution

### Performance Numbers to Highlight
- Cache hit startup: **~10ms per node** (instant)
- Cache miss startup: **~300ms per node** (one-time cost)
- Runtime overhead: **~1ms per node execution** (<0.1% of typical workflow)
- Memory: **Shared tensors, no duplication**

### Questions to Prepare For
1. **"Which node packs work on Cloud now?"**
   - Any node pack that doesn't use module-level web routes
   - See ComfyUI-APIsolated and ComfyUI-IsolationTest as proof

2. **"What's the performance impact?"**
   - See table above: ~1ms per execution, negligible for real workflows

3. **"Can users opt out?"**
   - Yes, just remove `pyisolate.yaml` or don't use `--use-process-isolation` flag

4. **"What about arbitrary code execution risks?"**
   - Current version: dependency isolation only
   - Future: filesystem/network sandboxing planned
   - Briefly show concept, but clarify it's not production-ready for untrusted code

---

**Ready to isolate?** Start with the [Quick Start](#installation) or check out the [Live Examples](#live-examples).

---

## Appendix: Supported APIs

### V1 `model_management` Functions (Proxied)

All standard model management functions are available in isolated nodes:

**Device Management:**
- `get_torch_device()` - Get the primary torch device
- `get_torch_device_name(device)` - Get device name string
- `unet_offload_device()` - Get offload device for models
- `vram_device()` - Get VRAM device
- `cpu_mode()` - Check if CPU-only mode
- `should_use_fp16()` - Check FP16 usage

**Memory Management:**
- `soft_empty_cache()` - Soft CUDA cache clear
- `unload_all_models()` - Unload all models from memory
- `get_free_memory(device)` - Get available memory
- `get_total_memory(device)` - Get total device memory

### V1 `folder_paths` Functions (Proxied)

All path resolution and file discovery functions work:

**Directory Access:**
- `get_input_directory()` - Get input folder path
- `get_output_directory()` - Get output folder path
- `get_temp_directory()` - Get temp folder path
- `get_folder_paths(folder_name)` - Get all paths for a folder type
- `models_dir` - Base models directory

**Model Discovery:**
- `get_filename_list(folder_name)` - List files in folder
- `get_full_path(folder_name, filename)` - Resolve full path
- `get_annotated_filepath(name)` - Get path with annotations
- `exists_annotated_filepath(name)` - Check annotated path exists

**Model Registration:**
- `add_model_folder_path(folder_name, full_folder_path)` - Register model folder
- `get_supported_pt_extensions()` - Get PyTorch file extensions

### V3 `comfy_api.latest` Support (Our current focus. Here is what's actually tested so far)

**✅ TESTED AND WORKING:**

**Node Base Classes:**
- `io.ComfyNode` - Base class for V3 nodes ✅
- `io.Input`, `io.WidgetInput`, `io.Output` - Input/output base classes ✅
- `io.Schema` - Node schema definition ✅

**Primitive Types (Fully Working):**
- `io.Boolean`, `io.Int`, `io.Float`, `io.String` ✅
- `io.Combo` ✅

**Data Types (Serializable - Fully Working):**
- `io.Image` - PyTorch tensor, zero-copy ✅
- `io.Mask` - PyTorch tensor, zero-copy ✅
- `io.Latent` - Dict with tensors ✅
- `io.Conditioning` - List of tuples with tensors ✅

**Execution API:**
- `ComfyAPI.execution.set_progress()` - Progress updates ✅

**Hidden Inputs:**
- `Hidden.unique_id`, `Hidden.prompt`, `Hidden.extra_pnginfo` ✅

---

**✅ DESIGNED TO WORK (V3 API Serialization-Friendly Types):**

The V3 API was designed with serialization in mind. Most types use pure data structures:

**Media & Tensors (All work via zero-copy):**
- `io.Audio` - Dict with waveform tensor + sample_rate ✅
- `io.Video` - Tensor-based ✅
- `io.SVG` - String data ✅

**Sampling & Noise:**
- `io.Sigmas` - Tensor ✅
- `io.Noise` - Serializable noise config ✅
- `io.TimestepsRange` - Simple data structure ✅

**3D & Analysis:**
- `io.Mesh`, `io.Voxel` - Data structures (vertices, faces, etc.) ✅
- `io.BBOX`, `io.Point`, `io.SEGS` - Coordinate data ✅
- `io.LossMap` - Tensor-based ✅

**Advanced (Likely work - need testing):**
- `io.ControlNet` - Might be serializable ⚠️
- `io.ClipVisionOutput` - Likely pure tensor output ⚠️
- `io.UpscaleModel`, `io.LoraModel` - Need testing ⚠️
- `io.Hooks`, `io.HookKeyframes` - Might be data structures ⚠️
- All `Load3D*` types - Need testing ⚠️

---

**❌ DEFINITIVELY DON'T WORK (Complex Objects):**

These are TYPE HINTS for non-serializable ComfyUI objects:
- `io.Model` - ModelPatcher object with methods/state ❌
- `io.Clip` - CLIP object with methods/state ❌
- `io.Vae` - VAE object with methods/state ❌
- `io.Sampler` - Sampler object with methods ❌
- `io.Guider` - CFGGuider object with methods ❌
- `io.ClipVision` - ClipVisionModel object ❌
- `io.StyleModel` - StyleModel object ❌
- `io.Gligen` - ModelPatcher variant ❌
- `io.AudioEncoder` - Encoder model object ❌
- `io.FaceAnalysis` - Analysis model object ❌

**Key insight:** The V3 API does ~80% of the heavy lifting. Most types are **designed** to be serializable (tensors, dicts, lists). Only the 8-10 "model object" types are problematic. `model_management` and `folder_paths` being fully proxied means V3 nodes get most ComfyUI functionality for free.

---

**❌ KNOWN NOT TO WORK:**

These require objects that cannot serialize across process boundaries:
- **ModelPatcher operations** - `model.clone()`, `model.add_wrapper_with_key()`, etc.
- **Direct Model/CLIP/VAE object manipulation** - These are complex stateful objects
- Any node that calls `model.model_options` directly

**Example that DOESN'T work:** The EasyCache node (attached) cannot be isolated because it requires direct ModelPatcher manipulation (`model.clone()`, `model.add_wrapper_with_key()`).

---

**Bottom line:** If your node uses **primitives, images, masks, latents, or conditioning**, it will likely work. If it touches **models, samplers, or advanced objects**, extensive testing is required.
