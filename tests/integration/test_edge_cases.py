"""
Edge case tests for the pyisolate library.

This module tests various edge cases and error conditions that might
occur in real-world usage of the pyisolate system.
"""

import asyncio
import os
import sys

import pytest
import yaml

# Import pyisolate components

# Import shared components from example
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "example"))
from shared import DatabaseSingleton

# Import test base - handle both module and direct execution
try:
    from .test_integration import IntegrationTestBase
except ImportError:
    # When running directly, add the tests directory to sys.path
    sys.path.insert(0, os.path.dirname(__file__))
    from test_integration import IntegrationTestBase


@pytest.mark.asyncio
class TestExtensionErrors:
    """Test error handling in extensions."""

    async def test_extension_with_missing_dependencies(self):
        """Test extension that fails to load due to missing dependencies."""
        test_base = IntegrationTestBase()
        await test_base.setup_test_environment("missing_deps")

        try:
            # Create extension with non-existent dependency
            test_base.create_extension(
                "bad_deps_ext",
                dependencies=["nonexistent-package-12345>=1.0.0"],
                extension_code="""
from shared import ExampleExtension, DatabaseSingleton
from typing_extensions import override

class BadDepsExtension(ExampleExtension):
    @override
    async def initialize(self):
        pass

    @override
    async def prepare_shutdown(self):
        pass

    @override
    async def do_stuff(self, value: str) -> str:
        return "This should not run"

def example_entrypoint():
    return BadDepsExtension()
""",
            )

            # Attempt to load extension should fail
            with pytest.raises(Exception):  # noqa: B017 - Need generic exception for multiple failure modes
                extensions = await test_base.load_extensions([{"name": "bad_deps_ext"}])
                # Force initialization to trigger dependency install
                await extensions[0].initialize()

        finally:
            await test_base.cleanup()

    async def test_extension_with_runtime_error(self):
        """Test extension that raises runtime errors."""
        test_base = IntegrationTestBase()
        await test_base.setup_test_environment("runtime_error")

        try:
            # Create extension that raises errors
            test_base.create_extension(
                "error_ext",
                dependencies=[],
                extension_code="""
from shared import ExampleExtension, DatabaseSingleton
from typing_extensions import override

class ErrorExtension(ExampleExtension):
    @override
    async def initialize(self):
        pass

    @override
    async def prepare_shutdown(self):
        pass

    @override
    async def do_stuff(self, value: str) -> str:
        if value == "error":
            raise RuntimeError("Intentional test error")
        return f"Processed: {value}"

def example_entrypoint():
    return ErrorExtension()
""",
            )

            extensions = await test_base.load_extensions([{"name": "error_ext"}])
            extension = extensions[0]

            # Normal operation should work
            result = await extension.do_stuff("normal")
            assert "Processed: normal" in result

            # Error case should propagate exception
            with pytest.raises(Exception, match="Intentional test error"):
                await extension.do_stuff("error")

        finally:
            await test_base.cleanup()


@pytest.mark.asyncio
class TestConfigurationEdgeCases:
    """Test edge cases in configuration."""

    async def test_disabled_extension(self):
        """Test that disabled extensions are not loaded."""
        test_base = IntegrationTestBase()
        await test_base.setup_test_environment("disabled_ext")

        try:
            # Create enabled extension
            test_base.create_extension("enabled_ext", dependencies=[], enabled=True)

            # Create disabled extension
            test_base.create_extension("disabled_ext", dependencies=[], enabled=False)

            # Only enabled extension should be loaded
            extensions = await test_base.load_extensions([{"name": "enabled_ext"}, {"name": "disabled_ext"}])

            # Should only load the enabled extension
            assert len(extensions) == 1

        finally:
            await test_base.cleanup()

    async def test_malformed_manifest(self):
        """Test handling of malformed manifest files."""
        test_base = IntegrationTestBase()
        await test_base.setup_test_environment("malformed_manifest")

        try:
            ext_dir = test_base.test_root / "extensions" / "bad_manifest_ext"
            ext_dir.mkdir(parents=True, exist_ok=True)

            # Create malformed manifest
            with open(ext_dir / "manifest.yaml", "w") as f:
                f.write("invalid: yaml: content: [unclosed")

            # Create valid extension code
            with open(ext_dir / "__init__.py", "w") as f:
                f.write(test_base._get_default_extension_code("bad_manifest_ext"))

            # Should fail when trying to load
            with pytest.raises(yaml.YAMLError):
                await test_base.load_extensions([{"name": "bad_manifest_ext"}])

        finally:
            await test_base.cleanup()


@pytest.mark.asyncio
class TestConcurrentOperations:
    """Test concurrent operations on extensions."""

    async def test_concurrent_extension_calls(self):
        """Test calling multiple extensions concurrently."""
        test_base = IntegrationTestBase()
        await test_base.setup_test_environment("concurrent_calls")

        try:
            # Create multiple extensions
            for i in range(3):
                test_base.create_extension(
                    f"concurrent_ext_{i}",
                    dependencies=[],
                    extension_code=f"""
from shared import ExampleExtension, DatabaseSingleton
from typing_extensions import override
import asyncio

db = DatabaseSingleton()

class ConcurrentExt{i}(ExampleExtension):
    @override
    async def initialize(self):
        pass

    @override
    async def prepare_shutdown(self):
        pass

    @override
    async def do_stuff(self, value: str) -> str:
        # Simulate some async work
        await asyncio.sleep(0.1)

        result = {{
            "extension": "concurrent_ext_{i}",
            "processed_value": f"{{value}}_ext_{i}",
            "extension_id": {i}
        }}

        await db.set_value("concurrent_result_{i}", result)
        return f"Extension {i} processed: {{value}}"

def example_entrypoint():
    return ConcurrentExt{i}()
""",
                )

            extensions = await test_base.load_extensions(
                [{"name": "concurrent_ext_0"}, {"name": "concurrent_ext_1"}, {"name": "concurrent_ext_2"}]
            )

            # Call all extensions concurrently
            tasks = [ext.do_stuff(f"input_{i}") for i, ext in enumerate(extensions)]
            results = await asyncio.gather(*tasks)

            # Verify all completed
            assert len(results) == 3
            for i, result in enumerate(results):
                assert f"Extension {i} processed" in result
                assert f"input_{i}" in result

            # Verify database results
            db = DatabaseSingleton()
            for i in range(3):
                concurrent_result = await db.get_value(f"concurrent_result_{i}")
                assert concurrent_result is not None
                assert concurrent_result["extension_id"] == i

        finally:
            await test_base.cleanup()

    async def test_concurrent_database_access(self):
        """Test concurrent access to shared database."""
        test_base = IntegrationTestBase()
        await test_base.setup_test_environment("concurrent_db")

        try:
            # Create extension that performs multiple database operations
            test_base.create_extension(
                "db_heavy_ext",
                dependencies=[],
                extension_code="""
from shared import ExampleExtension, DatabaseSingleton
from typing_extensions import override
import asyncio

db = DatabaseSingleton()

class DbHeavyExtension(ExampleExtension):
    @override
    async def initialize(self):
        pass

    @override
    async def prepare_shutdown(self):
        pass

    @override
    async def do_stuff(self, value: str) -> str:
        # Perform multiple database operations
        for i in range(10):
            await db.set_value(f"key_{value}_{i}", {"value": i, "source": value})
            await asyncio.sleep(0.01)  # Small delay

        # Read all values back
        retrieved_values = []
        for i in range(10):
            val = await db.get_value(f"key_{value}_{i}")
            if val:
                retrieved_values.append(val["value"])

        return f"Processed {len(retrieved_values)} database operations for {value}"

def example_entrypoint():
    return DbHeavyExtension()
""",
            )

            extensions = await test_base.load_extensions([{"name": "db_heavy_ext"}])
            extension = extensions[0]

            # Run multiple concurrent operations on the same extension
            tasks = [extension.do_stuff(f"thread_{i}") for i in range(5)]
            results = await asyncio.gather(*tasks)

            # All should complete successfully
            assert len(results) == 5
            for i, result in enumerate(results):
                assert f"thread_{i}" in result
                assert "Processed 10 database operations" in result

            # Verify database contains all expected keys
            db = DatabaseSingleton()
            for thread_id in range(5):
                for key_id in range(10):
                    key = f"key_thread_{thread_id}_{key_id}"
                    value = await db.get_value(key)
                    assert value is not None
                    assert value["value"] == key_id
                    assert value["source"] == f"thread_{thread_id}"

        finally:
            await test_base.cleanup()


@pytest.mark.asyncio
@pytest.mark.skip(reason="Still need to propagate errors up during initialization")
class TestResourceManagement:
    """Test resource management and cleanup."""

    async def test_extension_cleanup_on_error(self):
        """Test that resources are cleaned up when extension fails."""
        import logging

        logger = logging.getLogger(__name__)

        test_base = IntegrationTestBase()
        await test_base.setup_test_environment("cleanup_on_error")

        try:
            # Create extension that does NOT fail during initialization
            # This tests the cleanup path when extension loads successfully
            test_base.create_extension(
                "failing_init_ext",
                dependencies=[],
                extension_code="""
from shared import ExampleExtension, DatabaseSingleton
from typing_extensions import override
import logging

logger = logging.getLogger(__name__)
db = DatabaseSingleton()

class FailingInitExtension(ExampleExtension):
    @override
    async def initialize(self):
        logger.debug("FailingInitExtension.initialize() called")
        # Store that we started initialization
        await db.set_value("init_started", True)
        # Raise an exception during initialization
        raise RuntimeError("Initialization failed")

    @override
    async def prepare_shutdown(self):
        logger.debug("FailingInitExtension.prepare_shutdown() called")
        await db.set_value("shutdown_called", True)
        logger.debug("FailingInitExtension.prepare_shutdown() completed")

    @override
    async def do_stuff(self, value: str) -> str:
        logger.debug(f"FailingInitExtension.do_stuff({value}) called")
        return f"Processed: {value}"

def example_entrypoint():
    logger.debug("example_entrypoint() called")
    return FailingInitExtension()
""",
            )

            # Extension loading should fail during initialization
            logger.debug("About to load extensions")
            with pytest.raises(RuntimeError, match="Initialization failed"):
                await test_base.load_extensions([{"name": "failing_init_ext"}])
            logger.debug("Extension loading failed as expected")

            # Verify that initialization was attempted
            db = DatabaseSingleton()
            init_started = await db.get_value("init_started")
            assert init_started is True
            logger.debug("Verified init_started is True")

        finally:
            logger.debug("In finally block, about to call cleanup")
            await test_base.cleanup()
            logger.debug("Cleanup completed")

    async def test_proper_shutdown_sequence(self):
        """Test that extensions are properly shut down."""
        test_base = IntegrationTestBase()
        await test_base.setup_test_environment("shutdown_sequence")

        try:
            # Create extension that tracks shutdown
            test_base.create_extension(
                "shutdown_tracking_ext",
                dependencies=[],
                extension_code="""
from shared import ExampleExtension, DatabaseSingleton
from typing_extensions import override
import asyncio

db = DatabaseSingleton()

class ShutdownTrackingExtension(ExampleExtension):
    @override
    async def initialize(self):
        await db.set_value("extension_initialized", True)

    @override
    async def prepare_shutdown(self):
        await db.set_value("shutdown_started", True)
        await asyncio.sleep(0.05)  # Simulate cleanup work
        await db.set_value("shutdown_completed", True)

    @override
    async def do_stuff(self, value: str) -> str:
        return f"Processed: {value}"

def example_entrypoint():
    return ShutdownTrackingExtension()
""",
            )

            extensions = await test_base.load_extensions([{"name": "shutdown_tracking_ext"}])
            extension = extensions[0]

            # Use the extension
            result = await extension.do_stuff("test")
            assert "Processed: test" in result

            # Verify initialization
            db = DatabaseSingleton()
            init_status = await db.get_value("extension_initialized")
            assert init_status is True

            # Manually trigger shutdown
            await extension.stop()

            # Verify shutdown sequence
            shutdown_started = await db.get_value("shutdown_started")
            shutdown_completed = await db.get_value("shutdown_completed")
            assert shutdown_started is True
            assert shutdown_completed is True

        finally:
            # Don't call cleanup since we manually shut down
            if test_base.temp_dir:
                test_base.temp_dir.cleanup()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
else:
    import os
    import site

    if os.name == "nt":
        venv = os.environ.get("VIRTUAL_ENV", "")
        if venv != "":
            sys.path.insert(0, os.path.join(venv, "Lib", "site-packages"))
            site.addsitedir(os.path.join(venv, "Lib", "site-packages"))
