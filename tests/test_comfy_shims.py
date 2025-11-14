"""Tests for comfy helper shims (ProgressBar + common_upscale)."""

import importlib.util
import sys
from pathlib import Path

import torch

COMFY_ROOT = Path("/home/johnj/ComfyUI")
COMFY_PATH = COMFY_ROOT / "comfy" / "utils.py"
COMFY_ROOT.resolve()
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

spec = importlib.util.spec_from_file_location("comfy.utils", COMFY_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules["comfy.utils"] = module
assert spec.loader is not None
spec.loader.exec_module(module)  # type: ignore

ProgressBar = module.ProgressBar
common_upscale = module.common_upscale


def test_progress_bar_updates_locally():
    pb = ProgressBar(10)
    pb.update(3)
    pb.update_absolute(5, 10)
    state = pb.get_state()
    assert state["position"] == 5
    assert state["total"] == 10


def test_common_upscale_matches_expected_shape():
    tensor = torch.randn(1, 3, 16, 16)
    upscaled = common_upscale(tensor, 32, 32, "bilinear", "center")
    assert upscaled.shape == (1, 3, 32, 32)
