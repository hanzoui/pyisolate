
from pyisolate._internal import host


class FakeAdapter:
    identifier = "fake"

    def __init__(self, preferred_root="/tmp/ComfyUI"):
        self.preferred_root = preferred_root

    def get_path_config(self, module_path):
        return {
            "preferred_root": self.preferred_root,
            "additional_paths": [f"{self.preferred_root}/custom_nodes"],
        }

    def setup_child_environment(self, snapshot):
        return None

    def register_serializers(self, registry):
        return None

    def provide_rpc_services(self):
        return []

    def handle_api_registration(self, api, rpc):
        return None


def test_build_extension_snapshot_includes_adapter(monkeypatch):
    from pyisolate._internal.adapter_registry import AdapterRegistry
    monkeypatch.setattr(AdapterRegistry, "get", lambda: FakeAdapter())

    snapshot = host.build_extension_snapshot("/tmp/ComfyUI/custom_nodes/demo")

    assert "sys_path" in snapshot
    assert snapshot["adapter_name"] == "fake"
    assert snapshot["preferred_root"].endswith("ComfyUI")
    assert snapshot.get("additional_paths")
    assert snapshot.get("context_data", {}).get("module_path") == "/tmp/ComfyUI/custom_nodes/demo"


def test_build_extension_snapshot_no_adapter(monkeypatch):
    from pyisolate._internal.adapter_registry import AdapterRegistry
    monkeypatch.setattr(AdapterRegistry, "get", lambda: None)

    snapshot = host.build_extension_snapshot("/tmp/nowhere")
    assert "sys_path" in snapshot
    assert snapshot["adapter_name"] is None
    assert snapshot.get("preferred_root") is None
    assert snapshot.get("additional_paths") == []
