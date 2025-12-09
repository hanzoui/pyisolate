import asyncio
import importlib.util
import json
import logging
import os
import sys
import sysconfig
import warnings
from contextlib import nullcontext
from logging.handlers import QueueHandler
from pathlib import Path

from ..config import ExtensionConfig
from ..path_helpers import build_child_sys_path
from ..shared import ExtensionBase
from .shared import AsyncRPC, set_child_rpc_instance

logger = logging.getLogger(__name__)

if os.environ.get("PYISOLATE_CHILD") and os.environ.get("PYISOLATE_HOST_SNAPSHOT"):
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    logging.lastResort = None

    # Suppress ALL ComfyUI dependencies by loading requirements.txt
    import re
    snapshot = json.loads(Path(os.environ["PYISOLATE_HOST_SNAPSHOT"]).read_text())
    comfy_root = Path(snapshot.get("comfy_root", ""))
    requirements_path = comfy_root / "requirements.txt"

    for line in requirements_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Extract package name from requirements line (e.g., "torch>=1.0.0" -> "torch")
        pkg_name = re.split(r'[<>=!~\[]', line)[0].strip()
        if pkg_name:
            logging.getLogger(pkg_name).setLevel(logging.ERROR)

    class _ChildLogFilter(logging.Filter):
        _SUPPRESS = ("Total VRAM", "pytorch version:", "Set vram state to:",
                     "Device: cuda", "Enabled pinned memory", "Checkpoint files",
                     "working around nvidia", "Using pytorch attention")
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            return not any(p in msg for p in self._SUPPRESS)

    logging.getLogger().addFilter(_ChildLogFilter())
    warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*")

    snapshot_path = os.environ.get("PYISOLATE_HOST_SNAPSHOT")
    if snapshot_path and Path(snapshot_path).exists():
        with open(snapshot_path, "r") as f:
            snapshot = json.load(f)

        venv_site = sysconfig.get_path("purelib")
        venv_platlib = sysconfig.get_path("platlib")
        extra_paths = [venv_site, venv_platlib] if venv_site != venv_platlib else [venv_site]

        module_path = os.environ.get("PYISOLATE_MODULE_PATH", "")
        comfy_root = None
        if "ComfyUI" in module_path and "custom_nodes" in module_path:
            parts = module_path.split("ComfyUI")
            if len(parts) > 1:
                comfy_root = parts[0] + "ComfyUI"

        unified_path = build_child_sys_path(
            snapshot.get("sys_path", []),
            extra_paths,
            comfy_root=comfy_root
        )

        sys.path.clear()
        sys.path.extend(unified_path)

        import utils.json_util  # noqa: F401


async def async_entrypoint(
    module_path: str,
    extension_type: type[ExtensionBase],
    config: ExtensionConfig,
    to_extension,
    from_extension,
    log_queue,
) -> None:
    if os.environ.get("PYISOLATE_CHILD") and log_queue is not None:
        root = logging.getLogger()
        root.addHandler(QueueHandler(log_queue))
        root.setLevel(logging.INFO)

    rpc = AsyncRPC(recv_queue=to_extension, send_queue=from_extension)
    set_child_rpc_instance(rpc)

    extension = extension_type()
    extension._initialize_rpc(rpc)
    await extension.before_module_loaded()

    context = nullcontext()
    if config["share_torch"]:
        import torch
        context = torch.inference_mode()

    if not os.path.isdir(module_path):
        raise ValueError(f"Module path {module_path} is not a directory.")

    with context:
        rpc.register_callee(extension, "extension")
        for api in config["apis"]:
            api.use_remote(rpc)

            if api.__name__ == "LoggingRegistry":
                from comfy.isolation.proxies.logging_proxy import install_rpc_log_handler
                install_rpc_log_handler()

            if api.__name__ == "PromptServerProxy":
                import server
                proxy = api.instance
                original_register_route = proxy.register_route

                def register_route_wrapper(method, path, handler):
                    callback_id = rpc.register_callback(handler)
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(
                            original_register_route(method, path, handler=callback_id, is_callback=True))
                    else:
                        original_register_route(method, path, handler=callback_id, is_callback=True)
                    return None

                proxy.register_route = register_route_wrapper

                class RouteTableDefProxy:
                    def __init__(self, proxy_instance):
                        self.proxy = proxy_instance

                    def get(self, path, **kwargs):
                        def decorator(handler):
                            self.proxy.register_route("GET", path, handler)
                            return handler
                        return decorator

                    def post(self, path, **kwargs):
                        def decorator(handler):
                            self.proxy.register_route("POST", path, handler)
                            return handler
                        return decorator

                    def patch(self, path, **kwargs):
                        def decorator(handler):
                            self.proxy.register_route("PATCH", path, handler)
                            return handler
                        return decorator

                    def put(self, path, **kwargs):
                        def decorator(handler):
                            self.proxy.register_route("PUT", path, handler)
                            return handler
                        return decorator

                    def delete(self, path, **kwargs):
                        def decorator(handler):
                            self.proxy.register_route("DELETE", path, handler)
                            return handler
                        return decorator

                proxy.routes = RouteTableDefProxy(proxy)

                if hasattr(server, "PromptServer"):
                    if getattr(server.PromptServer, "instance", None) != proxy:
                        server.PromptServer.instance = proxy

        sys_module_name = os.path.basename(module_path).replace("-", "_").replace(".", "_")
        module_spec = importlib.util.spec_from_file_location(
            sys_module_name, os.path.join(module_path, "__init__.py")
        )

        assert module_spec is not None
        assert module_spec.loader is not None

        module = importlib.util.module_from_spec(module_spec)
        sys.modules[sys_module_name] = module
        module_spec.loader.exec_module(module)

        rpc.run()
        await extension.on_module_loaded(module)
        await rpc.run_until_stopped()


def entrypoint(
    module_path: str,
    extension_type: type[ExtensionBase],
    config: ExtensionConfig,
    to_extension,
    from_extension,
    log_queue,
) -> None:
    asyncio.run(
        async_entrypoint(
            module_path, extension_type, config,
            to_extension, from_extension, log_queue,
        )
    )
