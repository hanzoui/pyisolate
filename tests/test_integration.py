"""
Integration tests for the pyisolate library.

This test suite focuses on end-to-end testing of the pyisolate system,
testing multiple extensions with different dependencies, configurations,
and interaction patterns based on the example folder structure.
"""

import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TypedDict, cast

import pytest
import yaml

# Import pyisolate components
import pyisolate
from pyisolate import ExtensionConfig, ExtensionManager, ExtensionManagerConfig

# Import shared components from example
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "example"))
from shared import DatabaseSingleton, ExampleExtensionBase


class ManifestConfig(TypedDict):
    """Configuration structure for test manifests."""

    enabled: bool
    isolated: bool
    dependencies: list[str]
    share_torch: bool


class IntegrationTestBase:
    """Base class for integration tests providing common setup and utilities."""

    def __init__(self):
        self.temp_dir: Optional[tempfile.TemporaryDirectory] = None
        self.test_root: Optional[Path] = None
        self.manager: Optional[ExtensionManager] = None
        self.extensions: list[ExampleExtensionBase] = []

    async def setup_test_environment(self, test_name: str) -> Path:
        """Set up a temporary test environment."""
        # Create test directories within the project folder instead of system temp
        project_root = Path(__file__).parent.parent
        test_temps_dir = project_root / ".test_temps"
        test_temps_dir.mkdir(exist_ok=True)

        # Add timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.test_root = test_temps_dir / f"{test_name}_{timestamp}"
        self.test_root.mkdir(parents=True, exist_ok=True)

        # Store the path for cleanup
        self.temp_dir = None  # No longer using TemporaryDirectory

        # Create venv root directory
        venv_root = self.test_root / "extension-venvs"
        venv_root.mkdir(parents=True, exist_ok=True)

        # Create extensions directory
        extensions_dir = self.test_root / "extensions"
        extensions_dir.mkdir(parents=True, exist_ok=True)

        return self.test_root

    def create_extension(
        self,
        name: str,
        dependencies: list[str],
        share_torch: bool = False,
        isolated: bool = True,
        enabled: bool = True,
        extension_code: Optional[str] = None,
    ) -> Path:
        """Create a test extension with the given configuration."""
        if not self.test_root:
            raise RuntimeError("Test environment not set up")

        ext_dir = self.test_root / "extensions" / name
        ext_dir.mkdir(parents=True, exist_ok=True)

        # Create manifest.yaml
        manifest = {
            "enabled": enabled,
            "isolated": isolated,
            "dependencies": dependencies,
            "share_torch": share_torch,
        }

        with open(ext_dir / "manifest.yaml", "w") as f:
            yaml.dump(manifest, f)

        # Create __init__.py with extension code
        if extension_code is None:
            extension_code = self._get_default_extension_code(name)

        with open(ext_dir / "__init__.py", "w") as f:
            f.write(extension_code)

        return ext_dir

    def _get_default_extension_code(self, name: str) -> str:
        """Generate default extension code for testing."""
        return f'''
from shared import ExampleExtension, DatabaseSingleton
from typing_extensions import override
import logging

logger = logging.getLogger(__name__)
db = DatabaseSingleton()

class Test{name.capitalize()}(ExampleExtension):
    """Test extension {name}."""

    @override
    async def initialize(self):
        logger.debug("{name} initialized.")

    @override
    async def prepare_shutdown(self):
        logger.debug("{name} preparing for shutdown.")

    @override
    async def do_stuff(self, value: str) -> str:
        logger.debug(f"{name} processing: {{value}}")

        result = {{
            "extension": "{name}",
            "input_value": value,
            "processed": True
        }}

        await db.set_value("{name}_result", result)
        return f"{name} processed: {{value}}"

def example_entrypoint() -> ExampleExtension:
    """Entrypoint function for the extension."""
    return Test{name.capitalize()}()
'''

    async def load_extensions(self, extension_configs: list[dict[str, Any]]) -> list[ExampleExtensionBase]:
        """Load multiple extensions with given configurations."""
        logger = logging.getLogger(__name__)
        logger.debug(f"Starting to load {len(extension_configs)} extensions")

        if not self.test_root:
            raise RuntimeError("Test environment not set up")

        # Get pyisolate directory for editable install
        pyisolate_dir = os.path.dirname(os.path.dirname(os.path.realpath(pyisolate.__file__)))
        logger.debug(f"Pyisolate directory: {pyisolate_dir}")

        # Create extension manager
        config = ExtensionManagerConfig(venv_root_path=str(self.test_root / "extension-venvs"))
        logger.debug(f"Creating ExtensionManager with venv_root_path: {config['venv_root_path']}")
        self.manager = ExtensionManager(ExampleExtensionBase, config)

        extensions = []

        for idx, ext_config in enumerate(extension_configs):
            name = ext_config["name"]
            logger.debug(f"Loading extension {idx + 1}/{len(extension_configs)}: {name}")
            module_path = str(self.test_root / "extensions" / name)

            # Read manifest if not provided
            if "manifest" not in ext_config:
                yaml_path = Path(module_path) / "manifest.yaml"
                logger.debug(f"Reading manifest from: {yaml_path}")
                with open(yaml_path) as f:
                    manifest = cast(ManifestConfig, yaml.safe_load(f))
            else:
                manifest = ext_config["manifest"]

            if not manifest.get("enabled", True):
                logger.debug(f"Skipping disabled extension: {name}")
                continue

            # Create extension config
            extension_config = ExtensionConfig(
                name=name,
                module_path=module_path,
                isolated=manifest["isolated"],
                dependencies=manifest["dependencies"] + ["-e", pyisolate_dir],
                apis=[DatabaseSingleton],
                share_torch=manifest["share_torch"],
            )

            logger.debug(
                f"Loading extension with config: name={name}, isolated={manifest['isolated']}, "
                f"share_torch={manifest['share_torch']}, dependencies={manifest['dependencies']}"
            )

            extension = self.manager.load_extension(extension_config)
            logger.debug(f"Successfully loaded extension: {name}")
            extensions.append(extension)

        self.extensions = extensions
        logger.debug(f"Finished loading {len(extensions)} extensions")
        return extensions

    async def cleanup(self):
        """Clean up test environment."""
        # Shutdown all extensions via manager
        if self.manager:
            try:
                self.manager.stop_all_extensions()
            except Exception as e:
                logging.warning(f"Error stopping extensions: {e}")

        # Clean up test directory manually since we're not using TemporaryDirectory
        if self.test_root and self.test_root.exists():
            import shutil

            try:
                shutil.rmtree(self.test_root)
            except Exception as e:
                logging.warning(f"Error removing test directory {self.test_root}: {e}")


@pytest.mark.asyncio
class TestMultipleExtensionsWithConflictingDependencies:
    """Test loading multiple extensions with conflicting dependencies."""

    async def test_numpy_version_conflicts(self):
        """Test extensions with different numpy versions can coexist."""
        test_base = IntegrationTestBase()
        await test_base.setup_test_environment("numpy_conflicts")

        try:
            # Create extension with numpy 1.x
            test_base.create_extension(
                "numpy1_ext",
                dependencies=["numpy>=1.21.0,<2.0.0"],
                extension_code="""
from shared import ExampleExtension, DatabaseSingleton
from typing_extensions import override
import numpy as np
import logging

logger = logging.getLogger(__name__)
db = DatabaseSingleton()

class Numpy1Extension(ExampleExtension):
    @override
    async def initialize(self):
        logger.debug("Numpy1Extension initialized.")

    @override
    async def prepare_shutdown(self):
        logger.debug("Numpy1Extension preparing for shutdown.")

    @override
    async def do_stuff(self, value: str) -> str:
        version = np.__version__
        arr = np.array([1, 2, 3, 4, 5])
        result = {
            "extension": "numpy1_ext",
            "numpy_version": version,
            "array_sum": float(np.sum(arr)),
            "input_value": value
        }
        await db.set_value("numpy1_result", result)
        return f"Numpy1Extension processed with version {version}"

def example_entrypoint() -> ExampleExtension:
    return Numpy1Extension()
""",
            )

            # Create extension with numpy 2.x
            test_base.create_extension(
                "numpy2_ext",
                dependencies=["numpy>=2.0.0"],
                extension_code="""
from shared import ExampleExtension, DatabaseSingleton
from typing_extensions import override
import numpy as np
import logging

logger = logging.getLogger(__name__)
db = DatabaseSingleton()

class Numpy2Extension(ExampleExtension):
    @override
    async def initialize(self):
        logger.debug("Numpy2Extension initialized.")

    @override
    async def prepare_shutdown(self):
        logger.debug("Numpy2Extension preparing for shutdown.")

    @override
    async def do_stuff(self, value: str) -> str:
        version = np.__version__
        arr = np.array([2, 4, 6, 8, 10])
        result = {
            "extension": "numpy2_ext",
            "numpy_version": version,
            "array_sum": float(np.sum(arr)),
            "input_value": value
        }
        await db.set_value("numpy2_result", result)
        return f"Numpy2Extension processed with version {version}"

def example_entrypoint() -> ExampleExtension:
    return Numpy2Extension()
""",
            )

            # Load extensions
            extensions = await test_base.load_extensions([{"name": "numpy1_ext"}, {"name": "numpy2_ext"}])

            assert len(extensions) == 2

            # Execute extensions
            db = DatabaseSingleton()
            for ext in extensions:
                await ext.do_stuff("test_input")

            # Verify results
            numpy1_result = await db.get_value("numpy1_result")
            numpy2_result = await db.get_value("numpy2_result")

            assert numpy1_result is not None
            assert numpy2_result is not None
            assert numpy1_result["extension"] == "numpy1_ext"
            assert numpy2_result["extension"] == "numpy2_ext"
            assert numpy1_result["array_sum"] == 15.0  # 1+2+3+4+5
            assert numpy2_result["array_sum"] == 30.0  # 2+4+6+8+10

            # Verify versions are different major versions
            numpy1_version = numpy1_result["numpy_version"]
            numpy2_version = numpy2_result["numpy_version"]
            assert numpy1_version.startswith("1.")
            assert numpy2_version.startswith("2.")

        finally:
            await test_base.cleanup()


@pytest.mark.asyncio
class TestShareTorchConfiguration:
    """Test share_torch configuration scenarios."""

    async def test_share_torch_false(self):
        """Test extensions with share_torch=False have isolated torch."""
        test_base = IntegrationTestBase()
        await test_base.setup_test_environment("share_torch_false")

        try:
            # Create two extensions with torch, both with share_torch=False
            for i in [1, 2]:
                test_base.create_extension(
                    f"torch_ext_{i}",
                    dependencies=["torch>=1.9.0", "numpy>=2.0.0"],  # Add torch dependency
                    share_torch=False,
                    extension_code=f"""
from shared import ExampleExtension, DatabaseSingleton
from typing_extensions import override
import logging

logger = logging.getLogger(__name__)
db = DatabaseSingleton()

class TorchExt{i}(ExampleExtension):
    @override
    async def initialize(self):
        logger.debug("TorchExt{i} initialized.")

    @override
    async def prepare_shutdown(self):
        logger.debug("TorchExt{i} preparing for shutdown.")

    @override
    async def do_stuff(self, value: str) -> str:
        try:
            import torch
            tensor = torch.tensor([{i}.0, {i * 2}.0, {i * 3}.0])
            result = {{
                "extension": "torch_ext_{i}",
                "torch_available": True,
                "tensor_sum": float(torch.sum(tensor)),
                "torch_version": torch.__version__,
                "input_value": value
            }}
        except ImportError:
            result = {{
                "extension": "torch_ext_{i}",
                "torch_available": False,
                "input_value": value
            }}

        await db.set_value("torch_ext_{i}_result", result)
        return f"TorchExt{i} processed"

def example_entrypoint() -> ExampleExtension:
    return TorchExt{i}()
""",
                )

            # Load extensions
            extensions = await test_base.load_extensions([{"name": "torch_ext_1"}, {"name": "torch_ext_2"}])

            assert len(extensions) == 2

            # Execute extensions
            db = DatabaseSingleton()
            for ext in extensions:
                await ext.do_stuff("torch_test")

            # Verify results
            result1 = await db.get_value("torch_ext_1_result")
            result2 = await db.get_value("torch_ext_2_result")

            assert result1 is not None
            assert result2 is not None
            assert result1["torch_available"] is True
            assert result2["torch_available"] is True
            assert result1["tensor_sum"] == 6.0  # 1+2+3
            assert result2["tensor_sum"] == 12.0  # 2+4+6

        finally:
            await test_base.cleanup()

    async def test_share_torch_true(self):
        """Test extensions with share_torch=True share torch installation."""
        test_base = IntegrationTestBase()
        await test_base.setup_test_environment("share_torch_true")

        try:
            # Create two extensions with torch, both with share_torch=True
            for i in [1, 2]:
                test_base.create_extension(
                    f"shared_torch_ext_{i}",
                    dependencies=["torch>=1.9.0"],
                    share_torch=True,
                    extension_code=f"""
from shared import ExampleExtension, DatabaseSingleton
from typing_extensions import override
import logging

logger = logging.getLogger(__name__)
db = DatabaseSingleton()

class SharedTorchExt{i}(ExampleExtension):
    @override
    async def initialize(self):
        logger.debug("SharedTorchExt{i} initialized.")

    @override
    async def prepare_shutdown(self):
        logger.debug("SharedTorchExt{i} preparing for shutdown.")

    @override
    async def do_stuff(self, value: str) -> str:
        try:
            import torch
            tensor = torch.tensor([{i * 10}.0, {i * 20}.0])
            result = {{
                "extension": "shared_torch_ext_{i}",
                "torch_available": True,
                "tensor_sum": float(torch.sum(tensor)),
                "torch_version": torch.__version__,
                "input_value": value
            }}
        except ImportError:
            result = {{
                "extension": "shared_torch_ext_{i}",
                "torch_available": False,
                "input_value": value
            }}

        await db.set_value("shared_torch_ext_{i}_result", result)
        return f"SharedTorchExt{i} processed"

def example_entrypoint() -> ExampleExtension:
    return SharedTorchExt{i}()
""",
                )

            # Load extensions
            extensions = await test_base.load_extensions(
                [{"name": "shared_torch_ext_1"}, {"name": "shared_torch_ext_2"}]
            )

            assert len(extensions) == 2

            # Execute extensions
            db = DatabaseSingleton()
            for ext in extensions:
                await ext.do_stuff("shared_torch_test")

            # Verify results
            result1 = await db.get_value("shared_torch_ext_1_result")
            result2 = await db.get_value("shared_torch_ext_2_result")

            assert result1 is not None
            assert result2 is not None
            assert result1["torch_available"] is True
            assert result2["torch_available"] is True
            assert result1["tensor_sum"] == 30.0  # 10+20
            assert result2["tensor_sum"] == 60.0  # 20+40

            # Verify they're using the same torch version
            assert result1["torch_version"] == result2["torch_version"]

        finally:
            await test_base.cleanup()


@pytest.mark.asyncio
class TestHostExtensionInteraction:
    """Test calling between host and extensions."""

    async def test_host_calling_extension_functions(self):
        """Test host calling extension functions directly."""
        test_base = IntegrationTestBase()
        await test_base.setup_test_environment("host_to_extension")

        try:
            # Create extension with multiple methods
            test_base.create_extension(
                "multi_method_ext",
                dependencies=[],
                extension_code='''
from shared import ExampleExtension, DatabaseSingleton
from typing_extensions import override
import logging

logger = logging.getLogger(__name__)
db = DatabaseSingleton()

class MultiMethodExtension(ExampleExtension):
    def __init__(self):
        self.call_count = 0

    @override
    async def initialize(self):
        logger.debug("MultiMethodExtension initialized.")

    @override
    async def prepare_shutdown(self):
        logger.debug("MultiMethodExtension preparing for shutdown.")

    @override
    async def do_stuff(self, value: str) -> str:
        self.call_count += 1
        result = {
            "extension": "multi_method_ext",
            "method": "do_stuff",
            "call_count": self.call_count,
            "input_value": value
        }
        await db.set_value(f"do_stuff_call_{self.call_count}", result)
        return f"do_stuff processed: {value} (call #{self.call_count})"

    async def custom_method(self, data: dict) -> dict:
        """Custom method for testing host->extension calling."""
        self.call_count += 1
        result = {
            "extension": "multi_method_ext",
            "method": "custom_method",
            "call_count": self.call_count,
            "input_data": data,
            "processed_data": {**data, "processed": True}
        }
        await db.set_value(f"custom_method_call_{self.call_count}", result)
        return result["processed_data"]

def example_entrypoint() -> ExampleExtension:
    return MultiMethodExtension()
''',
            )

            # Load extension
            extensions = await test_base.load_extensions([{"name": "multi_method_ext"}])

            extension = extensions[0]
            db = DatabaseSingleton()

            # Test calling do_stuff method
            result1 = await extension.do_stuff("first_call")
            assert "first_call" in result1
            assert "call #1" in result1

            # Test calling do_stuff again
            result2 = await extension.do_stuff("second_call")
            assert "second_call" in result2
            assert "call #2" in result2

            # Verify database results
            call1_result = await db.get_value("do_stuff_call_1")
            call2_result = await db.get_value("do_stuff_call_2")

            assert call1_result["input_value"] == "first_call"
            assert call2_result["input_value"] == "second_call"
            assert call1_result["call_count"] == 1
            assert call2_result["call_count"] == 2

        finally:
            await test_base.cleanup()

    async def test_extension_calling_host_functions(self):
        """Test extensions calling host functions through shared APIs."""
        test_base = IntegrationTestBase()
        await test_base.setup_test_environment("extension_to_host")

        try:
            # Create extension that uses shared database extensively
            test_base.create_extension(
                "host_caller_ext",
                dependencies=[],
                extension_code="""
from shared import ExampleExtension, DatabaseSingleton
from typing_extensions import override
import logging

logger = logging.getLogger(__name__)
db = DatabaseSingleton()

class HostCallerExtension(ExampleExtension):
    @override
    async def initialize(self):
        logger.debug("HostCallerExtension initialized.")
        # Store initialization data
        await db.set_value("extension_initialized", {"status": "initialized", "extension": "host_caller_ext"})

    @override
    async def prepare_shutdown(self):
        logger.debug("HostCallerExtension preparing for shutdown.")
        await db.set_value("extension_shutdown", {"status": "shutting_down", "extension": "host_caller_ext"})

    @override
    async def do_stuff(self, value: str) -> str:
        # Use database to store intermediate results
        await db.set_value("processing_start", {"value": value, "step": "start"})

        # Simulate multi-step processing using host database
        for i in range(3):
            step_data = {"step": i+1, "value": f"{value}_step_{i+1}"}
            await db.set_value(f"processing_step_{i+1}", step_data)

        # Get back all the steps to verify host communication
        steps = []
        for i in range(3):
            step_result = await db.get_value(f"processing_step_{i+1}")
            if step_result:
                steps.append(step_result)

        final_result = {
            "extension": "host_caller_ext",
            "original_value": value,
            "steps_processed": len(steps),
            "final_value": f"{value}_processed"
        }

        await db.set_value("processing_complete", final_result)
        return f"Processed {value} through {len(steps)} steps"

def example_entrypoint() -> ExampleExtension:
    return HostCallerExtension()
""",
            )

            # Load extension
            extensions = await test_base.load_extensions([{"name": "host_caller_ext"}])

            extension = extensions[0]
            db = DatabaseSingleton()

            # Execute extension
            await extension.do_stuff("test_data")

            # Verify extension used host functions
            init_result = await db.get_value("extension_initialized")
            assert init_result["status"] == "initialized"

            processing_complete = await db.get_value("processing_complete")
            assert processing_complete["steps_processed"] == 3
            assert processing_complete["original_value"] == "test_data"

            # Verify all steps were stored
            for i in range(3):
                step_result = await db.get_value(f"processing_step_{i + 1}")
                assert step_result is not None
                assert step_result["step"] == i + 1

        finally:
            await test_base.cleanup()


@pytest.mark.asyncio
class TestRecursiveCalling:
    """Test recursive calling patterns between host and extensions."""

    async def test_host_extension_host_extension_calls(self):
        """Test recursive calls: host->extension->host->extension."""
        test_base = IntegrationTestBase()
        await test_base.setup_test_environment("recursive_calls")

        try:
            # Create extension that will trigger recursive calls
            test_base.create_extension(
                "recursive_ext",
                dependencies=[],
                extension_code="""
from shared import ExampleExtension, DatabaseSingleton
from typing_extensions import override
import logging

logger = logging.getLogger(__name__)
db = DatabaseSingleton()

class RecursiveExtension(ExampleExtension):
    def __init__(self):
        self.call_depth = 0

    @override
    async def initialize(self):
        logger.debug("RecursiveExtension initialized.")

    @override
    async def prepare_shutdown(self):
        logger.debug("RecursiveExtension preparing for shutdown.")

    @override
    async def do_stuff(self, value: str) -> str:
        self.call_depth += 1
        call_info = f"depth_{self.call_depth}"

        # Store call information in host database
        await db.set_value(f"call_{call_info}", {
            "depth": self.call_depth,
            "value": value,
            "caller": "extension"
        })

        # If we haven't reached max depth, trigger another level
        if self.call_depth < 3:
            # Store intermediate state
            await db.set_value(f"intermediate_{call_info}", {
                "about_to_recurse": True,
                "current_depth": self.call_depth
            })

            # Simulate calling back to host (through database interaction)
            # and then back to extension
            next_value = f"{value}_recursive_{self.call_depth}"

            # This simulates host processing
            await db.set_value(f"host_processing_{call_info}", {
                "processed_by": "host",
                "input": value,
                "output": next_value
            })

            # Now recurse (simulating host calling extension again)
            recursive_result = await self.do_stuff(next_value)

            final_result = f"Level{self.call_depth}: {value} -> {recursive_result}"
        else:
            final_result = f"MaxDepth{self.call_depth}: {value}"

        await db.set_value(f"result_{call_info}", {
            "depth": self.call_depth,
            "result": final_result
        })

        self.call_depth -= 1
        return final_result

def example_entrypoint() -> ExampleExtension:
    return RecursiveExtension()
""",
            )

            # Load extension
            extensions = await test_base.load_extensions([{"name": "recursive_ext"}])

            extension = extensions[0]
            db = DatabaseSingleton()

            # Trigger recursive calls
            result = await extension.do_stuff("initial")

            # Verify recursive call structure
            assert "Level1" in result
            assert "Level2" in result
            assert "MaxDepth3" in result

            # Verify all call levels were recorded
            for depth in [1, 2, 3]:
                call_result = await db.get_value(f"call_depth_{depth}")
                assert call_result is not None
                assert call_result["depth"] == depth

                result_data = await db.get_value(f"result_depth_{depth}")
                assert result_data is not None
                assert result_data["depth"] == depth

            # Verify intermediate host processing occurred
            for depth in [1, 2]:
                host_processing = await db.get_value(f"host_processing_depth_{depth}")
                assert host_processing is not None
                assert host_processing["processed_by"] == "host"

        finally:
            await test_base.cleanup()


@pytest.mark.asyncio
class TestComplexIntegrationScenarios:
    """Test complex scenarios combining multiple features."""

    async def test_multiple_extensions_with_cross_communication(self):
        """Test multiple extensions communicating through shared APIs."""
        test_base = IntegrationTestBase()
        await test_base.setup_test_environment("cross_communication")

        try:
            # Create producer extension
            test_base.create_extension(
                "producer_ext",
                dependencies=["numpy>=1.21.0"],
                extension_code="""
from shared import ExampleExtension, DatabaseSingleton
from typing_extensions import override
import numpy as np
import logging

logger = logging.getLogger(__name__)
db = DatabaseSingleton()

class ProducerExtension(ExampleExtension):
    @override
    async def initialize(self):
        logger.debug("ProducerExtension initialized.")

    @override
    async def prepare_shutdown(self):
        logger.debug("ProducerExtension preparing for shutdown.")

    @override
    async def do_stuff(self, value: str) -> str:
        # Generate some data
        data = np.random.rand(5).tolist()

        producer_result = {
            "extension": "producer_ext",
            "data": data,
            "data_sum": sum(data),
            "input_value": value
        }

        # Store for other extensions to consume
        await db.set_value("producer_data", producer_result)
        await db.set_value("data_ready", True)

        return f"Producer generated {len(data)} data points"

def example_entrypoint() -> ExampleExtension:
    return ProducerExtension()
""",
            )

            # Create consumer extension
            test_base.create_extension(
                "consumer_ext",
                dependencies=["scipy>=1.7.0"],
                extension_code="""
from shared import ExampleExtension, DatabaseSingleton
from typing_extensions import override
import scipy.stats as stats
import logging
import asyncio

logger = logging.getLogger(__name__)
db = DatabaseSingleton()

class ConsumerExtension(ExampleExtension):
    @override
    async def initialize(self):
        logger.debug("ConsumerExtension initialized.")

    @override
    async def prepare_shutdown(self):
        logger.debug("ConsumerExtension preparing for shutdown.")

    @override
    async def do_stuff(self, value: str) -> str:
        # Wait for producer data
        max_attempts = 10
        for attempt in range(max_attempts):
            data_ready = await db.get_value("data_ready")
            if data_ready:
                break
            await asyncio.sleep(0.1)

        producer_data = await db.get_value("producer_data")
        if not producer_data:
            return "No producer data available"

        # Process the data
        data = producer_data["data"]
        mean_val = stats.tmean(data)
        std_val = stats.tstd(data)

        consumer_result = {
            "extension": "consumer_ext",
            "consumed_data": data,
            "mean": float(mean_val),
            "std": float(std_val),
            "producer_sum": producer_data["data_sum"],
            "input_value": value
        }

        await db.set_value("consumer_result", consumer_result)

        return f"Consumer processed data: mean={mean_val:.3f}"

def example_entrypoint() -> ExampleExtension:
    return ConsumerExtension()
""",
            )

            # Load extensions
            extensions = await test_base.load_extensions([{"name": "producer_ext"}, {"name": "consumer_ext"}])

            assert len(extensions) == 2

            # Execute producer first, then consumer
            producer, consumer = extensions

            producer_result = await producer.do_stuff("produce_data")
            consumer_result = await consumer.do_stuff("consume_data")

            # Verify cross-communication worked
            db = DatabaseSingleton()
            producer_data = await db.get_value("producer_data")
            consumer_data = await db.get_value("consumer_result")

            assert producer_data is not None
            assert consumer_data is not None
            assert consumer_data["consumed_data"] == producer_data["data"]
            assert consumer_data["producer_sum"] == producer_data["data_sum"]

            assert "generated" in producer_result
            assert "processed data" in consumer_result

        finally:
            await test_base.cleanup()

    async def test_mixed_isolation_and_sharing(self):
        """Test mix of isolated and non-isolated extensions."""
        test_base = IntegrationTestBase()
        await test_base.setup_test_environment("mixed_isolation")

        try:
            # Create isolated extension
            test_base.create_extension(
                "isolated_ext",
                dependencies=["requests>=2.25.0"],
                isolated=True,
                extension_code="""
from shared import ExampleExtension, DatabaseSingleton
from typing_extensions import override
import logging

logger = logging.getLogger(__name__)
db = DatabaseSingleton()

class IsolatedExtension(ExampleExtension):
    @override
    async def initialize(self):
        logger.debug("IsolatedExtension initialized.")

    @override
    async def prepare_shutdown(self):
        logger.debug("IsolatedExtension preparing for shutdown.")

    @override
    async def do_stuff(self, value: str) -> str:
        try:
            import requests
            # Don't actually make HTTP request in tests
            result = {
                "extension": "isolated_ext",
                "isolation": "isolated",
                "requests_available": True,
                "input_value": value
            }
        except ImportError:
            result = {
                "extension": "isolated_ext",
                "isolation": "isolated",
                "requests_available": False,
                "input_value": value
            }

        await db.set_value("isolated_result", result)
        return "Isolated extension processed"

def example_entrypoint() -> ExampleExtension:
    return IsolatedExtension()
""",
            )

            # Create non-isolated extension
            test_base.create_extension(
                "shared_ext",
                dependencies=[],
                isolated=False,
                extension_code="""
from shared import ExampleExtension, DatabaseSingleton
from typing_extensions import override
import sys
import logging

logger = logging.getLogger(__name__)
db = DatabaseSingleton()

class SharedExtension(ExampleExtension):
    @override
    async def initialize(self):
        logger.debug("SharedExtension initialized.")

    @override
    async def prepare_shutdown(self):
        logger.debug("SharedExtension preparing for shutdown.")

    @override
    async def do_stuff(self, value: str) -> str:
        # This extension shares the host environment
        result = {
            "extension": "shared_ext",
            "isolation": "shared",
            "python_path": sys.path[:3],  # First few entries
            "input_value": value
        }

        await db.set_value("shared_result", result)
        return "Shared extension processed"

def example_entrypoint() -> ExampleExtension:
    return SharedExtension()
""",
            )

            # Load extensions
            extensions = await test_base.load_extensions([{"name": "isolated_ext"}, {"name": "shared_ext"}])

            assert len(extensions) == 2

            # Execute both extensions
            for ext in extensions:
                await ext.do_stuff("mixed_test")

            # Verify results
            db = DatabaseSingleton()
            isolated_result = await db.get_value("isolated_result")
            shared_result = await db.get_value("shared_result")

            assert isolated_result is not None
            assert shared_result is not None
            assert isolated_result["isolation"] == "isolated"
            assert shared_result["isolation"] == "shared"

        finally:
            await test_base.cleanup()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
else:
    import os
    import site

    if os.name == "nt":
        venv = os.environ.get("VIRTUAL_ENV", "")
        if venv != "":
            # Add virtual environment site-packages to sys.path
            sys.path.insert(0, os.path.join(venv, "Lib", "site-packages"))
            site.addsitedir(os.path.join(venv, "Lib", "site-packages"))
