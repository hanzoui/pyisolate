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

logger = logging.getLogger(__name__)


# Apply host sys.path snapshot immediately on module import if we're a PyIsolate child
# This must happen BEFORE any other ComfyUI imports during multiprocessing spawn
if os.environ.get("PYISOLATE_CHILD") == "1":
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
            
            # Diagnostic prints (logging not configured yet during spawn)
            print(f"📚 [PyIsolate][PathUnification] sys.path unification completed", file=sys.stderr)
            print(f"📚 [PyIsolate][PathUnification] comfy_root={comfy_root}", file=sys.stderr)
            print(f"📚 [PyIsolate][PathUnification] ComfyUI in sys.path: {comfy_root in unified_path if comfy_root else 'N/A'}", file=sys.stderr)
            print(f"📚 [PyIsolate][PathUnification] First 5 unified paths: {unified_path[:5]}", file=sys.stderr)
            print(f"📚 [PyIsolate][PathUnification] Total unified paths: {len(unified_path)}", file=sys.stderr)
            
            # Replace sys.path
            sys.path.clear()
            sys.path.extend(unified_path)
            
            logger.info(
                "📚 [PyIsolate][Client] Applied host snapshot on module import (comfy_root=%s, paths=%d)",
                comfy_root,
                len(unified_path)
            )
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
    logger.info(
        "📚 [PyIsolate][Client] Starting async_entrypoint module_path=%s executable=%s share_torch=%s",
        module_path,
        sys.executable,
        config["share_torch"],
    )

    rpc = AsyncRPC(recv_queue=to_extension, send_queue=from_extension)
    extension = extension_type()
    extension._initialize_rpc(rpc)
    await extension.before_module_loaded()

    context = nullcontext()
    if config["share_torch"]:
        import torch

        context = torch.inference_mode()

    if not os.path.isdir(module_path):
        msg = f"Module path {module_path} is not a directory."
        logger.error("�📚 [PyIsolate][Client] %s", msg)
        raise ValueError(msg)

    with context:
        try:
            rpc.register_callee(extension, "extension")
            for api in config["apis"]:
                api.use_remote(rpc)

            sys_module_name = module_path.replace(".", "_x_")
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

                logger.error("�📚 [PyIsolate][Client] on_module_loaded failed for %s: %s", module_path, e)
                logger.error("Exception details:\n%s", traceback.format_exc())
                await rpc.stop()
                raise

            await rpc.run_until_stopped()

        except Exception as e:
            import traceback

            logger.error("�📚 [PyIsolate][Client] Error loading extension from %s: %s", module_path, e)
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
