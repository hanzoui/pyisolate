import json
import sys

import pytest

from pyisolate._internal import bootstrap
from pyisolate._internal.serialization_registry import SerializerRegistry


class FakeAdapter:
    identifier = "fake"

    def __init__(self):
        self.setup_called = False
        self.registry_used = False

    def get_path_config(self, module_path):
        return None

    def setup_child_environment(self, snapshot):
        self.setup_called = True

    def register_serializers(self, registry):
        self.registry_used = True
        registry.register("FakeType", lambda x: {"v": x}, lambda x: x["v"])

    def provide_rpc_services(self):
        return []

    def handle_api_registration(self, api, rpc):
        return None


@pytest.fixture(autouse=True)
def clear_registry():
    registry = SerializerRegistry.get_instance()
    registry.clear()
    yield
    registry.clear()


def test_bootstrap_applies_snapshot(monkeypatch, tmp_path):
    fake_adapter = FakeAdapter()
    monkeypatch.setattr(bootstrap, "_rehydrate_adapter", lambda name: fake_adapter)

    snapshot = {
        "sys_path": [str(tmp_path / "foo")],
        "adapter_ref": "fake:FakeAdapter",
    }
    monkeypatch.setenv("PYISOLATE_HOST_SNAPSHOT", json.dumps(snapshot))

    original_sys_path = list(sys.path)
    try:
        adapter = bootstrap.bootstrap_child()
        updated_sys_path = list(sys.path)
    finally:
        sys.path[:] = original_sys_path

    assert adapter is fake_adapter
    assert fake_adapter.setup_called
    assert fake_adapter.registry_used
    assert snapshot["sys_path"][0] in updated_sys_path

    registry = SerializerRegistry.get_instance()
    assert registry.has_handler("FakeType")


def test_bootstrap_no_snapshot(monkeypatch):
    monkeypatch.delenv("PYISOLATE_HOST_SNAPSHOT", raising=False)
    assert bootstrap.bootstrap_child() is None


def test_bootstrap_bad_json(monkeypatch):
    monkeypatch.setenv("PYISOLATE_HOST_SNAPSHOT", "not-json")
    with pytest.raises(ValueError):
        bootstrap.bootstrap_child()


def test_bootstrap_missing_adapter(monkeypatch):
    monkeypatch.setenv("PYISOLATE_HOST_SNAPSHOT", json.dumps({"adapter_ref": "missing"}))
    monkeypatch.setattr(
        bootstrap, "_rehydrate_adapter", lambda name: (_ for _ in ()).throw(ValueError("nope"))
    )
    with pytest.raises(ValueError):
        bootstrap.bootstrap_child()
