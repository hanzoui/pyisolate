"""
Pytest configuration and fixtures.

Add any shared fixtures or pytest configuration here.
"""

import logging
import os
import sys

# Add ComfyUI to sys.path BEFORE any tests run
# This is required because pyisolate is now ComfyUI-integrated
COMFYUI_ROOT = "/home/johnj/ComfyUI"
if COMFYUI_ROOT not in sys.path:
    sys.path.insert(0, COMFYUI_ROOT)

# Set environment variable so child processes know ComfyUI location
os.environ.setdefault("COMFYUI_ROOT", COMFYUI_ROOT)


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Set up logging
    log_level = logging.DEBUG if config.getoption("--debug-pyisolate") else logging.INFO

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Set specific logger levels
    logging.getLogger("pyisolate").setLevel(log_level)
    logging.getLogger("asyncio").setLevel(log_level)

    # If custom log file is specified, add file handler
    custom_log_file = config.getoption("--pyisolate-log-file")
    if custom_log_file:
        file_handler = logging.FileHandler(custom_log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
            )
        )
        logging.getLogger().addHandler(file_handler)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--debug-pyisolate",
        action="store_true",
        default=False,
        help="Enable debug logging for pyisolate (shows detailed execution flow)",
    )
    parser.addoption(
        "--pyisolate-log-file",
        action="store",
        default=None,
        help="Log pyisolate debug output to specified file",
    )
