"""Minimal ExtensionBase implementation for the hello world sample."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict

from pyisolate import ExtensionBase

try:
    from .custom_nodes.simple_text_node.shared_services import HelloWorldModelManager
except ImportError:  # pragma: no cover - exercised manually
    import sys
    from pathlib import Path

    _NODE_DIR = Path(__file__).parent / "custom_nodes" / "simple_text_node"
    sys.path.insert(0, str(_NODE_DIR))
    from shared_services import HelloWorldModelManager  # type: ignore

logger = logging.getLogger(__name__)


class ComfyNodeExtension(ExtensionBase):
    """Wraps NODE_CLASS_MAPPINGS-based custom nodes behind RPC calls."""

    def __init__(self) -> None:
        super().__init__()
        self.node_classes: Dict[str, type] = {}
        self.node_instances: Dict[str, object] = {}
        self.display_names: Dict[str, str] = {}

    async def on_module_loaded(self, module):  # pragma: no cover - exercised manually
        self.node_classes = getattr(module, "NODE_CLASS_MAPPINGS", {}) or {}
        if not self.node_classes:
            raise RuntimeError("Module missing NODE_CLASS_MAPPINGS")

        self.display_names = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {}) or {}
        self.node_instances = {name: node_cls() for name, node_cls in self.node_classes.items()}
        self.register_callee(self, "extension")
        logger.info("Hello world extension loaded with %d node(s)", len(self.node_instances))

    async def list_nodes(self) -> Dict[str, str]:
        return {name: self.display_names.get(name, name) for name in self.node_classes}

    async def get_node_info(self, node_name: str) -> Dict[str, Any]:
        node_cls = self._get_node_class(node_name)
        return {
            "input_types": node_cls.INPUT_TYPES() if hasattr(node_cls, "INPUT_TYPES") else {},
            "return_types": getattr(node_cls, "RETURN_TYPES", ()),
            "function": getattr(node_cls, "FUNCTION", "process"),
            "category": getattr(node_cls, "CATEGORY", ""),
            "output_node": getattr(node_cls, "OUTPUT_NODE", False),
        }

    async def execute_node(self, node_name: str, **inputs: Any):
        instance = self._get_node_instance(node_name)
        node_cls = self._get_node_class(node_name)
        fn_name = getattr(node_cls, "FUNCTION", "process")
        method = getattr(instance, fn_name)
        result = method(**inputs)
        if inspect.iscoroutine(result):
            result = await result
        if not isinstance(result, tuple):
            result = (result,)
        return result

    async def fetch_default_model_name(self) -> str:
        manager = HelloWorldModelManager()
        return await manager.get_default_model_name()

    def _get_node_class(self, node_name: str) -> type:
        if node_name not in self.node_classes:
            raise KeyError(f"Unknown node: {node_name}")
        return self.node_classes[node_name]

    def _get_node_instance(self, node_name: str) -> object:
        if node_name not in self.node_instances:
            raise KeyError(f"Node {node_name} has not been instantiated")
        return self.node_instances[node_name]
