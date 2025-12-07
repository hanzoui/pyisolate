#!/usr/bin/env python3
"""CLI tool to verify route extraction before loading.

Usage:
    python -m comfy.isolation.tools.extract_routes <node_path> [--output FILE]
    
Examples:
    python -m comfy.isolation.tools.extract_routes /path/to/custom_nodes/cg-image-filter
    python -m comfy.isolation.tools.extract_routes ./custom_nodes/comfyui-ollama --output routes.json

Rev 1.0: Per PYISOLATE_COMFY_INTEGRATION_ARCHITECTURE.md Section 12.2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract HTTP routes from ComfyUI custom node for PyIsolate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m comfy.isolation.tools.extract_routes ./custom_nodes/cg-image-filter
    python -m comfy.isolation.tools.extract_routes /path/to/node --output manifest.json
    python -m comfy.isolation.tools.extract_routes ./node --verbose
        """
    )
    
    parser.add_argument(
        "node_path",
        type=Path,
        help="Path to custom node directory"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file path (default: <node_path>/route_manifest.json)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing manifest, don't generate new one"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON only (for scripting)"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    node_path = args.node_path.resolve()
    
    if not node_path.exists():
        print(f" Error: Path does not exist: {node_path}", file=sys.stderr)
        return 1
    
    if not node_path.is_dir():
        print(f" Error: Path is not a directory: {node_path}", file=sys.stderr)
        return 1
    
    # Import route_extractor directly (it has no heavy dependencies)
    # We need to avoid importing __init__.py which pulls in pyisolate
    # Strategy: Add parent path to sys.path and import by file path directly
    
    import importlib.util
    
    route_extractor_path = Path(__file__).parent.parent / "route_extractor.py"
    
    spec = importlib.util.spec_from_file_location(
        "route_extractor",
        route_extractor_path
    )
    route_extractor = importlib.util.module_from_spec(spec)
    
    # CRITICAL: Set __name__ before exec_module to fix dataclass decorator
    route_extractor.__name__ = "route_extractor"
    sys.modules["route_extractor"] = route_extractor
    
    spec.loader.exec_module(route_extractor)
    
    extract_routes = route_extractor.extract_routes
    generate_route_manifest = route_extractor.generate_route_manifest
    validate_route_manifest = route_extractor.validate_route_manifest
    
    # Validate-only mode
    if args.validate_only:
        manifest_path = args.output or (node_path / "route_manifest.json")
        is_valid, errors = validate_route_manifest(manifest_path)
        
        if args.json:
            print(json.dumps({"valid": is_valid, "errors": errors}))
        else:
            if is_valid:
                print(f" Manifest is valid: {manifest_path}")
            else:
                print(f" Manifest validation failed:")
                for error in errors:
                    print(f"   - {error}")
        
        return 0 if is_valid else 1
    
    # Extract routes
    routes = extract_routes(node_path)
    
    if args.json:
        # JSON output mode for scripting
        manifest = {
            "node_name": node_path.name,
            "version": "1.0",
            "routes": [r.to_dict() for r in routes]
        }
        print(json.dumps(manifest, indent=2))
        return 0
    
    # Human-readable output
    print(f"\n][ Route Extraction Report")
    print(f"{'=' * 50}")
    print(f"Node:   {node_path.name}")
    print(f"Path:   {node_path}")
    print(f"Routes: {len(routes)}")
    print()
    
    if not routes:
        print("   (No routes found)")
        print()
        print("  This node has no HTTP routes to inject.")
        print("   If you expected routes, check that decorators use the pattern:")
        print("   @PromptServer.instance.routes.post('/path')")
        return 0
    
    # Print route table
    print(f"{'Method':<8} {'Path':<40} {'Handler':<30}")
    print(f"{'-' * 8} {'-' * 40} {'-' * 30}")
    
    for route in routes:
        handler = f"{route.handler_module}.{route.handler_func}"
        if len(handler) > 30:
            handler = handler[:27] + "..."
        print(f"{route.method:<8} {route.path:<40} {handler:<30}")
    
    print()
    
    # Generate manifest if output specified
    if args.output:
        manifest = generate_route_manifest(node_path, args.output)
        print(f" Manifest written to: {args.output}")
    else:
        # Show manifest preview
        print("Manifest JSON Preview:")
        print("-" * 50)
        manifest = {
            "node_name": node_path.name,
            "version": "1.0",
            "routes": [r.to_dict() for r in routes]
        }
        print(json.dumps(manifest, indent=2))
        print()
        print(f"  To save: python -m comfy.isolation.tools.extract_routes {node_path} --output route_manifest.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
