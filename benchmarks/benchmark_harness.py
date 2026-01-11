import os
import sys
import shutil
import tempfile
import asyncio
from pathlib import Path
from typing import Optional, Any
from contextlib import contextmanager

from pyisolate import ExtensionManagerConfig, ExtensionManager, ExtensionConfig

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class BenchmarkHarness:
    """Harness for running benchmarks without depending on test suite infrastructure."""

    def __init__(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="pyisolate_bench_")
        self.test_root = Path(self.temp_dir.name)
        (self.test_root / "extensions").mkdir(exist_ok=True)
        (self.test_root / "extension-venvs").mkdir(exist_ok=True)
        self.extensions = []
        self.manager = None

    async def setup_test_environment(self, name: str) -> None:
        """Initialize the benchmark environment."""
        # Ensure uv is in PATH (required for venv creation)
        venv_bin = os.path.dirname(sys.executable)
        path = os.environ.get("PATH", "")
        if venv_bin not in path.split(os.pathsep):
            os.environ["PATH"] = f"{venv_bin}{os.pathsep}{path}"

        # Setup shared temp for Torch file_system IPC
        # This is CRITICAL for share_torch=True to work in sandboxed environments
        shared_tmp = self.test_root / "ipc_shared"
        shared_tmp.mkdir(parents=True, exist_ok=True)
        # Force host process (and children via inherit) to use this TMPDIR
        os.environ["TMPDIR"] = str(shared_tmp)
        
        print(f"Benchmark Harness initialized at {self.test_root}")
        print(f"IPC Shared Directory: {shared_tmp}")

        # Ensure proper torch multiprocessing setup
        if TORCH_AVAILABLE:
            try:
                import torch.multiprocessing
                torch.multiprocessing.set_sharing_strategy('file_system')
            except ImportError:
                pass


    def create_extension(
        self,
        name: str,
        dependencies: list[str],
        share_torch: bool,
        extension_code: str
    ) -> None:
        """Create an extension module on disk."""
        ext_dir = self.test_root / "extensions" / name
        ext_dir.mkdir(parents=True, exist_ok=True)
        (ext_dir / "__init__.py").write_text(extension_code)

    async def load_extensions(self, extension_configs: list[dict], extension_base_cls) -> list:
        """Load extensions defined in configs."""
        config = ExtensionManagerConfig(venv_root_path=str(self.test_root / "extension-venvs"))
        self.manager = ExtensionManager(extension_base_cls, config)
        
        loaded_extensions = []
        for cfg in extension_configs:
            name = cfg["name"]
            # Config might be passed as simple dict
            
            # Reconstruct dependencies if not passed mostly for existing pattern in benchmark.py
            # But create_extension handles writing to disk. loading needs ExtensionConfig object.
            
            # This is slightly tricky because creation and loading are split in benchmark.py
            # I'll rely on the caller to pass correct params or infer them?
            # Actually benchmark.py logic: create_extension then load_extensions loop.
            
             # Since we know the path structure from create_extension:
            module_path = str(self.test_root / "extensions" / name)
            
            # NOTE: benchmark.py passed deps to create_extension but strangely not to load_extensions
            # We must pass them here to ExtensionConfig. 
            # Ideally load_extensions accepts full config objects or we recreate them.
            # I will adapt this to match what benchmark.py expects or refactor benchmark.py to iterate.
            
            # Simpler approach: Allow caller to just use manager directly if they want, 
            # or provide a helper that does what benchmark.py did (but correctly).
            pass
            
        return loaded_extensions # placeholder, I will implement explicit loading in the script

    def get_manager(self, extension_base_cls):
        if not self.manager:
             config = ExtensionManagerConfig(venv_root_path=str(self.test_root / "extension-venvs"))
             self.manager = ExtensionManager(extension_base_cls, config)
        return self.manager

    async def cleanup(self):
        """Clean up resources."""
        if self.manager:
            try:
                self.manager.stop_all_extensions()
            except Exception as e:
                print(f"Error stopping extensions: {e}")
                
        if self.temp_dir:
            self.temp_dir.cleanup()
