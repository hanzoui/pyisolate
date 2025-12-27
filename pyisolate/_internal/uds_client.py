"""Entry point for isolated child processes (JSON-RPC).

This module is invoked by the host process as `python -m pyisolate._internal.uds_client`.
It connects to the host via Unix Domain Socket (UDS) using JSON-RPC,
receives bootstrap configuration, and delegates to the standard async_entrypoint.

This replaces the old pickle-based client.py entrypoint.

Environment variables expected:
- PYISOLATE_UDS_ADDRESS: Path to the Unix socket to connect to
- PYISOLATE_CHILD: Set to "1" by the host
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import socket
import sys
from contextlib import AbstractContextManager as ContextManager
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ..config import ExtensionConfig

from .tensor_serializer import register_tensor_serializer

logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for isolated child processes."""
    # 1. Get UDS address from environment
    uds_address = os.environ.get("PYISOLATE_UDS_ADDRESS")
    if not uds_address:
        raise RuntimeError(
            "PYISOLATE_UDS_ADDRESS not set. "
            "This module should only be invoked via host launcher."
        )

    # 2. Connect to host via UDS (raw socket for JSON-RPC)
    logger.info("[PyIsolate][JSON-RPC] Connecting to host at %s", uds_address)
    client_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client_sock.connect(uds_address)

    # 3. Create JSON transport (NO PICKLE)
    from .shared import JSONSocketTransport
    transport = JSONSocketTransport(client_sock)

    # 4. Receive bootstrap data from host via JSON
    bootstrap_data = transport.recv()
    logger.info("[PyIsolate][JSON-RPC] Received bootstrap data")

    # 5. Apply host snapshot to environment
    snapshot = bootstrap_data.get("snapshot", {})
    os.environ["PYISOLATE_HOST_SNAPSHOT"] = json.dumps(snapshot)
    os.environ["PYISOLATE_CHILD"] = "1"

    # 6. Bootstrap the child environment (apply sys.path, etc.)
    from .bootstrap import bootstrap_child
    bootstrap_child()

    # 7. Import remaining dependencies after bootstrap
    from ..shared import ExtensionBase

    # 8. Extract configuration from bootstrap data
    config: ExtensionConfig = bootstrap_data["config"]
    module_path: str = config["module_path"]

    # Extension type is serialized as "module.classname" string reference
    ext_type_ref = bootstrap_data.get("extension_type_ref", "pyisolate.shared.ExtensionBase")

    # Resolve extension type from string reference
    try:
        parts = ext_type_ref.rsplit(".", 1)
        if len(parts) == 2:
            import importlib
            module = importlib.import_module(parts[0])
            extension_type = getattr(module, parts[1])
        else:
            extension_type = ExtensionBase
    except Exception as e:
        logger.warning(
            "[PyIsolate][JSON-RPC] Could not resolve extension type %s: %s",
            ext_type_ref, e
        )
        extension_type = ExtensionBase

    # 9. Run the async entrypoint
    asyncio.run(_async_uds_entrypoint(
        transport=transport,
        module_path=module_path,
        extension_type=extension_type,
        config=config,
    ))


async def _async_uds_entrypoint(
    transport: Any,
    module_path: str,
    extension_type: type[Any],
    config: ExtensionConfig,
) -> None:
    """Async entrypoint for isolated processes using JSON-RPC transport."""
    from ..interfaces import IsolationAdapter
    from .loader import load_adapter
    from .shared import (
        AsyncRPC,
        ProxiedSingleton,
        set_child_rpc_instance,
    )

    # RPC uses the existing JSONSocketTransport
    rpc = AsyncRPC(transport=transport)
    set_child_rpc_instance(rpc)

    # Register tensor serializer
    from .serialization_registry import SerializerRegistry
    register_tensor_serializer(SerializerRegistry.get_instance())

    # Ensure file_system strategy for CPU tensors
    import torch
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Instantiate extension
    extension = extension_type()
    extension._initialize_rpc(rpc)

    try:
        await extension.before_module_loaded()
    except Exception as exc:
        logger.error("Extension before_module_loaded failed: %s", exc, exc_info=True)
        raise

    # Set up torch inference mode if share_torch enabled
    context: ContextManager[Any] = nullcontext()
    if config.get("share_torch", False):
        import torch
        context = cast(ContextManager[Any], torch.inference_mode())

    if not os.path.isdir(module_path):
        raise ValueError(f"Module path {module_path} is not a directory.")

    # Load adapter for API registration
    adapter: IsolationAdapter | None = None
    with contextlib.suppress(Exception):
        adapter = load_adapter()

    # CRITICAL: Register serializers in child process
    if adapter:
        from .serialization_registry import SerializerRegistry
        adapter.register_serializers(SerializerRegistry.get_instance())

    with context:
        rpc.register_callee(extension, "extension")

        # Register APIs from config
        apis = config.get("apis", [])
        resolved_apis = []

        # Resolve string references back to classes if needed
        for api_item in apis:
            if isinstance(api_item, str):
                try:
                    import importlib
                    parts = api_item.rsplit(".", 1)
                    if len(parts) == 2:
                        mod = importlib.import_module(parts[0])
                        resolved_apis.append(getattr(mod, parts[1]))
                    else:
                        logger.warning("Invalid API reference format: %s", api_item)
                except Exception as e:
                    logger.warning("Failed to resolve API %s: %s", api_item, e)
            else:
                resolved_apis.append(api_item)

        for api in resolved_apis:
            api.use_remote(rpc)
            if adapter:
                api_instance = cast(ProxiedSingleton, getattr(api, "instance", api))
                logger.info("[UDS] Calling handle_api_registration for %s", api_instance.__class__.__name__)
                adapter.handle_api_registration(api_instance, rpc)
                # Verify UtilsProxy specifically
                if api_instance.__class__.__name__ == "UtilsProxy":
                    import comfy.utils
                    logger.info("[UDS] After UtilsProxy registration: PROGRESS_BAR_HOOK = %s", 
                               comfy.utils.PROGRESS_BAR_HOOK)

        # Import and load the extension module
        import importlib.util
        sys_module_name = os.path.basename(module_path).replace("-", "_").replace(".", "_")
        module_spec = importlib.util.spec_from_file_location(
            sys_module_name, os.path.join(module_path, "__init__.py")
        )

        assert module_spec is not None
        assert module_spec.loader is not None

        try:
            # Check PROGRESS_BAR_HOOK before module load
            try:
                import comfy.utils
                logger.info("[UDS] BEFORE module load: PROGRESS_BAR_HOOK = %s", comfy.utils.PROGRESS_BAR_HOOK)
            except Exception:
                pass
            
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[sys_module_name] = module
            module_spec.loader.exec_module(module)
            
            # Check PROGRESS_BAR_HOOK after module load
            try:
                import comfy.utils
                logger.info("[UDS] AFTER module load: PROGRESS_BAR_HOOK = %s", comfy.utils.PROGRESS_BAR_HOOK)
            except Exception:
                pass

            rpc.run()
            await extension.on_module_loaded(module)
            await rpc.run_until_stopped()
        except Exception as exc:
            logger.error(
                "Extension module loading/execution failed for %s: %s",
                module_path, exc, exc_info=True
            )
            raise


if __name__ == "__main__":
    main()
