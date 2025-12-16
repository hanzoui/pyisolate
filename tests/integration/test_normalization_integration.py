"""Integration tests for extension name normalization."""

import os

# Import test base
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_integration import IntegrationTestBase


@pytest.mark.asyncio
class TestExtensionNameNormalization:
    """Test that extension names with spaces and special characters work correctly."""

    async def test_extension_with_spaces(self):
        """Test that extensions with spaces in names work correctly."""
        test_base = IntegrationTestBase()
        await test_base.setup_test_environment("name_normalization")

        try:
            # Create extension with spaces in name
            test_base.create_extension(
                "my cool extension",
                dependencies=[],
                extension_code="""
from shared import ExampleExtension, DatabaseSingleton
from typing_extensions import override
import logging

logger = logging.getLogger(__name__)
db = DatabaseSingleton()

class SpacedExtension(ExampleExtension):
    @override
    async def initialize(self):
        logger.debug("Extension with spaces initialized.")
        await db.set_value("init_name", self.__class__.__module__)

    @override
    async def prepare_shutdown(self):
        logger.debug("Extension with spaces shutting down.")

    @override
    async def do_stuff(self, value: str) -> str:
        return f"Processed by extension with spaces: {value}"

def example_entrypoint() -> ExampleExtension:
    return SpacedExtension()
""",
            )

            # Load the extension
            extensions = await test_base.load_extensions([{"name": "my cool extension"}])
            assert len(extensions) == 1

            # Test that it works
            result = await extensions[0].do_stuff("test")
            assert "Processed by extension with spaces" in result

            # Verify the venv was created with normalized name
            venv_root = test_base.test_root / "extension-venvs"
            normalized_venv = venv_root / "my_cool_extension"
            assert normalized_venv.exists()

            # Original name with spaces should NOT exist
            spaces_venv = venv_root / "my cool extension"
            assert not spaces_venv.exists()

        finally:
            await test_base.cleanup()

    async def test_extension_with_unicode(self):
        """Test that extensions with Unicode names work correctly."""
        test_base = IntegrationTestBase()
        await test_base.setup_test_environment("unicode_normalization")

        try:
            # Create extension with Unicode name
            test_base.create_extension(
                "扩展 extension",  # Chinese + English with space
                dependencies=[],
                extension_code="""
from shared import ExampleExtension, DatabaseSingleton
from typing_extensions import override

class UnicodeExtension(ExampleExtension):
    @override
    async def initialize(self):
        pass

    @override
    async def prepare_shutdown(self):
        pass

    @override
    async def do_stuff(self, value: str) -> str:
        return f"Unicode extension processed: {value}"

def example_entrypoint() -> ExampleExtension:
    return UnicodeExtension()
""",
            )

            # Load and test
            extensions = await test_base.load_extensions([{"name": "扩展 extension"}])
            result = await extensions[0].do_stuff("测试")
            assert "Unicode extension processed" in result

            # Check normalized path preserves Unicode but replaces space
            venv_root = test_base.test_root / "extension-venvs"
            normalized_venv = venv_root / "扩展_extension"
            assert normalized_venv.exists()

        finally:
            await test_base.cleanup()

    async def test_extension_with_dangerous_chars(self):
        """Test that extensions with potentially dangerous characters are normalized."""
        test_base = IntegrationTestBase()
        await test_base.setup_test_environment("dangerous_chars")

        try:
            # Create extension with shell metacharacters
            test_base.create_extension(
                "ext$(echo test)",
                dependencies=[],
                extension_code="""
from shared import ExampleExtension, DatabaseSingleton
from typing_extensions import override

class SafeExtension(ExampleExtension):
    @override
    async def initialize(self):
        pass

    @override
    async def prepare_shutdown(self):
        pass

    @override
    async def do_stuff(self, value: str) -> str:
        return f"Safe extension processed: {value}"

def example_entrypoint() -> ExampleExtension:
    return SafeExtension()
""",
            )

            # Should work with normalized name
            extensions = await test_base.load_extensions([{"name": "ext$(echo test)"}])
            result = await extensions[0].do_stuff("test")
            assert "Safe extension processed" in result

            # Check the venv has safe name
            venv_root = test_base.test_root / "extension-venvs"
            normalized_venv = venv_root / "ext_echo_test"
            assert normalized_venv.exists()

        finally:
            await test_base.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
