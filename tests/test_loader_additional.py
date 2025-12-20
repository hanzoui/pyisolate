import importlib.metadata as importlib_metadata
from types import SimpleNamespace

import pytest

from pyisolate._internal import loader


class DummyEntryPoint:
    def __init__(self, name, obj=None):
        self.name = name
        self._obj = obj or SimpleNamespace()

    def load(self):
        return lambda: self._obj


def _fake_eps(eps_list):
    """Create a fake entry_points function."""
    return lambda: SimpleNamespace(select=lambda group=None: eps_list)


def test_load_adapter_override_env(monkeypatch):
    monkeypatch.setenv("PYISOLATE_ADAPTER_OVERRIDE", "demo")
    eps = [DummyEntryPoint("demo")]
    monkeypatch.setattr(importlib_metadata, "entry_points", _fake_eps(eps))

    adapter = loader.load_adapter()
    assert isinstance(adapter, SimpleNamespace)


def test_load_adapter_not_found_raises(monkeypatch):
    monkeypatch.delenv("PYISOLATE_ADAPTER_OVERRIDE", raising=False)
    eps = []
    monkeypatch.setattr(importlib_metadata, "entry_points", _fake_eps(eps))
    with pytest.raises(ValueError):
        loader.load_adapter("missing")


def test_load_adapter_multiple_available_requires_name(monkeypatch):
    eps = [DummyEntryPoint("a"), DummyEntryPoint("b")]
    monkeypatch.setattr(importlib_metadata, "entry_points", _fake_eps(eps))
    with pytest.raises(ValueError):
        loader.load_adapter()


def test_load_adapter_specific_name(monkeypatch):
    eps = [DummyEntryPoint("x"), DummyEntryPoint("target", obj={"ok": True})]
    monkeypatch.setattr(importlib_metadata, "entry_points", _fake_eps(eps))
    adapter = loader.load_adapter("target")
    assert adapter == {"ok": True}


def test_load_adapter_autodetect_single(monkeypatch):
    eps = [DummyEntryPoint("solo", obj={"auto": True})]
    monkeypatch.setattr(importlib_metadata, "entry_points", _fake_eps(eps))
    adapter = loader.load_adapter()
    assert adapter == {"auto": True}
