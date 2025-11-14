"""Simple ComfyUI-style node used by the hello world example."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .shared_services import HelloWorldModelManager


class SimpleTextNode:
    """Toy node that uppercases text and reports requests' version."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, tuple]:  # pragma: no cover - tiny demo
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "default": "HELLO FROM COMFYUI WITH PYISOLATE!",
                        "multiline": True,
                    },
                )
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "pyisolate/example"
    OUTPUT_NODE = False

    def __init__(self) -> None:
        # Pretend to keep a handle to a shared service; real nodes will lazily
        # resolve these when they actually need them.
        self._model_manager = HelloWorldModelManager()

    def process(self, text: str):  # pragma: no cover - exercised manually
        import requests

        requests_version = requests.__version__
        emphasized = np.char.upper(text)
        message = f"[Processed with requests {requests_version}]: {emphasized}"
        return (message,)


NODE_CLASS_MAPPINGS = {
    "SimpleTextNode": SimpleTextNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleTextNode": "PyIsolate Simple Text",
}
