from __future__ import annotations

import importlib
from typing import Any


def get_torch_optional() -> tuple[Any | None, Any | None]:
    """Return (torch, torch.multiprocessing.reductions) when available.

    PyTorch is optional for base pyisolate usage. Callers that need tensor
    features should use `require_torch(...)` for explicit errors.
    """
    try:
        torch = importlib.import_module("torch")
        reductions = importlib.import_module("torch.multiprocessing.reductions")
        return torch, reductions
    except Exception:
        return None, None


def require_torch(feature_name: str) -> tuple[Any, Any]:
    """Return torch modules or raise a clear feature-scoped error."""
    torch, reductions = get_torch_optional()
    if torch is None or reductions is None:
        raise RuntimeError(f"{feature_name} requires PyTorch. Install 'torch' to use this feature.")
    return torch, reductions
