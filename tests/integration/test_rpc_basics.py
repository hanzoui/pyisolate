"""Integration tests for PyIsolate RPC and extension mechanisms."""

import asyncio
import time
from typing import Any

import pytest

from pyisolate import ExtensionBase, ExtensionManager, ExtensionManagerConfig
from pyisolate.config import ExtensionConfig


class DummyExtension(ExtensionBase):
    """Minimal extension for testing RPC mechanics."""
    
    def __init__(self):
        super().__init__()
        self.call_count = 0
    
    async def on_module_loaded(self, module: Any) -> None:
        """Called after module import."""
        pass
    
    async def echo(self, message: str) -> str:
        """Simple RPC method - echo back the message."""
        self.call_count += 1
        return f"Echo: {message}"
    
    async def add_numbers(self, a: int, b: int) -> int:
        """Test method with multiple args."""
        return a + b
    
    async def raise_error(self, message: str) -> None:
        """Test error propagation."""
        raise ValueError(f"Intentional error: {message}")
    
    async def get_call_count(self) -> int:
        """Get number of times methods were called."""
        return self.call_count


@pytest.mark.asyncio
class TestRPCBasics:
    """Test basic RPC functionality without complex setup."""
    
    async def test_extension_manager_loads_dummy_extension(self, tmp_path):
        """Extension manager can load a minimal extension."""
        # Create a dummy module
        module_dir = tmp_path / "dummy_module"
        module_dir.mkdir()
        (module_dir / "__init__.py").write_text("# Dummy module\n")
        
        venv_root = tmp_path / "venvs"
        venv_root.mkdir()
        
        manager_config = ExtensionManagerConfig(venv_root_path=str(venv_root))
        manager = ExtensionManager(DummyExtension, manager_config)
        
        config = ExtensionConfig(
            name="dummy",
            module_path=str(module_dir),
            isolated=True,
            dependencies=[],
            apis=[],
            share_torch=False,
        )
        
        extension = manager.load_extension(config)
        
        assert extension is not None
        await manager.unload_all()
    
    async def test_rpc_echo_call(self, tmp_path):
        """Can call a simple RPC method and get response."""
        module_dir = tmp_path / "dummy_module"
        module_dir.mkdir()
        (module_dir / "__init__.py").write_text("# Dummy module\n")
        
        venv_root = tmp_path / "venvs"
        venv_root.mkdir()
        
        manager_config = ExtensionManagerConfig(venv_root_path=str(venv_root))
        manager = ExtensionManager(DummyExtension, manager_config)
        
        config = ExtensionConfig(
            name="dummy",
            module_path=str(module_dir),
            isolated=True,
            dependencies=[],
            apis=[],
            share_torch=False,
        )
        
        extension = manager.load_extension(config)
        
        result = await extension.echo("test message")
        assert result == "Echo: test message"
        
        await manager.unload_all()
    
    async def test_rpc_multiple_calls(self, tmp_path):
        """Multiple RPC calls work correctly."""
        module_dir = tmp_path / "dummy_module"
        module_dir.mkdir()
        (module_dir / "__init__.py").write_text("# Dummy module\n")
        
        venv_root = tmp_path / "venvs"
        venv_root.mkdir()
        
        manager_config = ExtensionManagerConfig(venv_root_path=str(venv_root))
        manager = ExtensionManager(DummyExtension, manager_config)
        
        config = ExtensionConfig(
            name="dummy",
            module_path=str(module_dir),
            isolated=True,
            dependencies=[],
            apis=[],
            share_torch=False,
        )
        
        extension = manager.load_extension(config)
        
        # Call multiple times
        result1 = await extension.echo("first")
        result2 = await extension.echo("second")
        result3 = await extension.add_numbers(10, 20)
        
        assert result1 == "Echo: first"
        assert result2 == "Echo: second"
        assert result3 == 30
        
        # Verify call counting works
        count = await extension.get_call_count()
        assert count == 3  # echo, echo, add_numbers
        
        await manager.unload_all()
    
    async def test_rpc_error_propagation(self, tmp_path):
        """Errors in isolated process propagate to host."""
        module_dir = tmp_path / "dummy_module"
        module_dir.mkdir()
        (module_dir / "__init__.py").write_text("# Dummy module\n")
        
        venv_root = tmp_path / "venvs"
        venv_root.mkdir()
        
        manager_config = ExtensionManagerConfig(venv_root_path=str(venv_root))
        manager = ExtensionManager(DummyExtension, manager_config)
        
        config = ExtensionConfig(
            name="dummy",
            module_path=str(module_dir),
            isolated=True,
            dependencies=[],
            apis=[],
            share_torch=False,
        )
        
        extension = manager.load_extension(config)
        
        # Should raise ValueError
        with pytest.raises(Exception) as exc_info:
            await extension.raise_error("test error")
        
        assert "Intentional error: test error" in str(exc_info.value)
        
        await manager.unload_all()


@pytest.mark.asyncio
class TestRPCPerformance:
    """Test RPC overhead and batching."""
    
    async def test_rpc_call_overhead(self, tmp_path):
        """Measure RPC overhead for rapid calls."""
        module_dir = tmp_path / "dummy_module"
        module_dir.mkdir()
        (module_dir / "__init__.py").write_text("# Dummy module\n")
        
        venv_root = tmp_path / "venvs"
        venv_root.mkdir()
        
        manager_config = ExtensionManagerConfig(venv_root_path=str(venv_root))
        manager = ExtensionManager(DummyExtension, manager_config)
        
        config = ExtensionConfig(
            name="dummy",
            module_path=str(module_dir),
            isolated=True,
            dependencies=[],
            apis=[],
            share_torch=False,
        )
        
        extension = manager.load_extension(config)
        
        # Warm up
        await extension.echo("warmup")
        
        # Time 100 calls
        start = time.perf_counter()
        for i in range(100):
            await extension.add_numbers(i, i + 1)
        duration = time.perf_counter() - start
        
        avg_ms = (duration / 100) * 1000
        
        # RPC overhead should be < 5ms per call (generous for CI)
        assert avg_ms < 5.0, f"RPC overhead too high: {avg_ms:.2f}ms per call"
        
        await manager.unload_all()
    
    async def test_rpc_concurrent_calls(self, tmp_path):
        """Multiple concurrent RPC calls work correctly."""
        module_dir = tmp_path / "dummy_module"
        module_dir.mkdir()
        (module_dir / "__init__.py").write_text("# Dummy module\n")
        
        venv_root = tmp_path / "venvs"
        venv_root.mkdir()
        
        manager_config = ExtensionManagerConfig(venv_root_path=str(venv_root))
        manager = ExtensionManager(DummyExtension, manager_config)
        
        config = ExtensionConfig(
            name="dummy",
            module_path=str(module_dir),
            isolated=True,
            dependencies=[],
            apis=[],
            share_torch=False,
        )
        
        extension = manager.load_extension(config)
        
        # Launch 10 concurrent calls
        tasks = [
            extension.add_numbers(i, i * 2)
            for i in range(10)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all results correct
        expected = [i + (i * 2) for i in range(10)]
        assert results == expected
        
        await manager.unload_all()
