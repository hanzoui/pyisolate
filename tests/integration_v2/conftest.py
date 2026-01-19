import pytest

from tests.harness.host import ReferenceHost


@pytest.fixture
async def reference_host():
    """Provides a ReferenceHost instance."""
    host = ReferenceHost()
    host.setup()
    try:
        yield host
    finally:
        await host.cleanup()
