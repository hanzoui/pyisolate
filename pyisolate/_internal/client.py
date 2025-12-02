import asyncio
import importlib.util
import json
import logging
import os
import os.path
import sys
import sysconfig
from contextlib import nullcontext
from pathlib import Path

from ..config import ExtensionConfig
from ..path_helpers import build_child_sys_path
from ..shared import ExtensionBase
from .shared import AsyncRPC

# Configure child process logging to forward to parent's stdout
# This must happen BEFORE any logging occurs in the child
if os.environ.get("PYISOLATE_CHILD"):
    logging.basicConfig(
        level=logging.INFO,
        format='📚 %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout,
        force=True
    )

logger = logging.getLogger(__name__)
PATH_LOGGING_ENABLED = bool(os.environ.get("PYISOLATE_PATH_DEBUG"))


def _log_paths(unified_path: list[str]) -> None:
    if not PATH_LOGGING_ENABLED:
        return
    logger.debug(
        "📚 [PyIsolate][PathUnification] sys.path (%d entries):\n%s",
        len(unified_path),
        "\n".join(f"  [{i}] {p}" for i, p in enumerate(unified_path))
    )


# Apply host sys.path snapshot immediately on module import if we're a PyIsolate child
# This must happen BEFORE any other ComfyUI imports during multiprocessing spawn
if os.environ.get("PYISOLATE_CHILD"):
    # Configure logging FIRST - before any other imports that might log
    logging.basicConfig(
        level=logging.INFO,
        format='📚 %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout,
        force=True  # Override any existing config
    )
    
    # Suppress noisy ComfyUI startup logs in child processes
    # These have zero informational value when repeated per-child
    class _ChildLogFilter(logging.Filter):
        """Filter out noisy ComfyUI startup messages in child processes."""
        _SUPPRESS_PATTERNS = (
            "Total VRAM",
            "pytorch version:",
            "Set vram state to:",
            "Device: cuda",
            "Enabled pinned memory",
            "Checkpoint files will always",
            "working around nvidia",
            "Using pytorch attention",
        )
        
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            if any(pattern in msg for pattern in self._SUPPRESS_PATTERNS):
                # Emit dot to stderr to show life without noise
                sys.stderr.write(".")
                sys.stderr.flush()
                return False
            return True
    
    logging.getLogger().addFilter(_ChildLogFilter())
    
    # Suppress pynvml deprecation warnings in child processes
    import warnings
    warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*")
    
    snapshot_path = os.environ.get("PYISOLATE_HOST_SNAPSHOT")
    if snapshot_path and Path(snapshot_path).exists():
        try:
            with open(snapshot_path, "r") as f:
                snapshot = json.load(f)
            
            # Get isolated venv site-packages
            venv_site = sysconfig.get_path("purelib")
            venv_platlib = sysconfig.get_path("platlib")
            extra_paths = [venv_site, venv_platlib] if venv_site != venv_platlib else [venv_site]
            
            # Detect ComfyUI root from PYISOLATE_MODULE_PATH
            module_path = os.environ.get("PYISOLATE_MODULE_PATH", "")
            comfy_root = None
            if "ComfyUI" in module_path and "custom_nodes" in module_path:
                parts = module_path.split("ComfyUI")
                if len(parts) > 1:
                    comfy_root = parts[0] + "ComfyUI"
            
            # Build unified sys.path
            unified_path = build_child_sys_path(
                snapshot.get("sys_path", []),
                extra_paths,
                comfy_root=comfy_root
            )
            _log_paths(unified_path)
            sys.path.clear()
            sys.path.extend(unified_path)
            
            # Test utils import immediately after path configuration
            try:
                import utils.json_util  # noqa: F401
            except Exception as import_err:
                if PATH_LOGGING_ENABLED:
                    logger.debug(
                        "📚 [PyIsolate][PathUnification] ❌ utils.json_util import FAILED: %s",
                        import_err,
                    )
                try:
                    import utils as utils_test
                except Exception as utils_err:
                    if PATH_LOGGING_ENABLED:
                        logger.debug(
                            "📚 [PyIsolate][PathUnification] 'utils' import also failed: %s",
                            utils_err,
                        )
                    raise
                raise

        except Exception as e:
            logger.error("📚 [PyIsolate][Client] Failed to apply host snapshot on import: %s", e)
            raise


async def async_entrypoint(
    module_path: str,
    extension_type: type[ExtensionBase],
    config: ExtensionConfig,
    to_extension,
    from_extension,
) -> None:
    """
    Asynchronous entrypoint for the module.
    """
    rpc = AsyncRPC(recv_queue=to_extension, send_queue=from_extension)
    
    # Store RPC globally for deserialization use
    from .shared import set_child_rpc_instance
    set_child_rpc_instance(rpc)
    
    extension = extension_type()
    extension._initialize_rpc(rpc)
    await extension.before_module_loaded()

    context = nullcontext()
    if config["share_torch"]:
        import torch

        context = torch.inference_mode()

    if not os.path.isdir(module_path):
        msg = f"Module path {module_path} is not a directory."
        logger.error("📚 [PyIsolate][Client] %s", msg)
        raise ValueError(msg)

    with context:
        try:
            rpc.register_callee(extension, "extension")
            for api in config["apis"]:
                api.use_remote(rpc)

                # Install RPC log handler if this is LoggingRegistry
                if api.__name__ == "LoggingRegistry":
                    try:
                        from comfy.isolation.proxies.logging_proxy import install_rpc_log_handler
                        install_rpc_log_handler()
                    except Exception as e:
                        logger.warning(f"📚 [PyIsolate][Client] Failed to install RPC log handler: {e}")

                # Patch PromptServer.instance.register_route if it's the PromptServer proxy
                if api.__name__ == "PromptServerProxy":
                    try:
                        import server

                        proxy = api.instance
                        original_register_route = proxy.register_route

                        def register_route_wrapper(method, path, handler):
                            callback_id = rpc.register_callback(handler)

                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                asyncio.create_task(
                                    original_register_route(
                                        method, path, handler=callback_id, is_callback=True
                                    )
                                )
                            else:
                                original_register_route(
                                    method, path, handler=callback_id, is_callback=True
                                )
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
                    except Exception as e:
                        logger.warning(f"📚 [PyIsolate][Client] Failed to patch PromptServer: {e}")

            # Use just the directory name as the module name to avoid paths in __module__
            # This prevents pickle errors when classes are serialized across processes
            sys_module_name = os.path.basename(module_path).replace("-", "_").replace(".", "_")
            module_spec = importlib.util.spec_from_file_location(
                sys_module_name, os.path.join(module_path, "__init__.py")
            )

            assert module_spec is not None, f"Module spec for {module_path} is None"
            assert module_spec.loader is not None, f"Module loader for {module_path} is None"

            module = importlib.util.module_from_spec(module_spec)
            sys.modules[sys_module_name] = module

            module_spec.loader.exec_module(module)

            rpc.run()
            try:
                await extension.on_module_loaded(module)
            except Exception as e:
                import traceback

                logger.error("📚 [PyIsolate][Client] on_module_loaded failed for %s: %s", module_path, e)
                logger.error("Exception details:\n%s", traceback.format_exc())
                await rpc.stop()
                raise

            await rpc.run_until_stopped()

        except Exception as e:
            import traceback

            logger.error("📚 [PyIsolate][Client] Error loading extension from %s: %s", module_path, e)
            logger.error("Exception details:\n%s", traceback.format_exc())
            raise


def entrypoint(
    module_path: str,
    extension_type: type[ExtensionBase],
    config: ExtensionConfig,
    to_extension,
    from_extension,
) -> None:
    asyncio.run(async_entrypoint(module_path, extension_type, config, to_extension, from_extension))
