"""Shared services used by the ComfyUI hello world example.

These classes live alongside the custom node so the host and the isolated
process import the exact same definitions. The example uses a
``ProxiedSingleton`` so the isolated node can talk back to the host process
and request data (mimicking how we will proxy ComfyUI globals).
"""

from __future__ import annotations

from pyisolate import ProxiedSingleton


class HelloWorldModelManager(ProxiedSingleton):
    """Minimal stand-in for ComfyUI's folder/model managers."""

    def __init__(self) -> None:
        super().__init__()
        self._default_model = "example/StableDream-v1.safetensors"

    async def get_default_model_name(self) -> str:
        """Return a pretend model path to prove RPC wiring works."""

        return self._default_model
