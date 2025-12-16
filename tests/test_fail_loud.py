import pytest

from pyisolate._internal import bootstrap, loader


def test_bootstrap_missing_adapter_fails(monkeypatch):
    monkeypatch.setenv("PYISOLATE_HOST_SNAPSHOT", '{"adapter_name": "missing"}')
    monkeypatch.setattr(bootstrap, "load_adapter", lambda name: (_ for _ in ()).throw(ValueError("missing")))

    with pytest.raises(ValueError):
        bootstrap.bootstrap_child()


def test_load_adapter_ambiguous(monkeypatch):
    class DummyEP:
        def __init__(self, name):
            self.name = name
        def load(self):
            return object

    class EPObj:
        def __init__(self, eps):
            self._eps = eps
        def select(self, **kwargs):
            return self._eps

    eps = [DummyEP("a"), DummyEP("b")]

    def fake_entry_points():
        return EPObj(eps)

    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    monkeypatch.setattr(loader, "entry_points", fake_entry_points, raising=False)

    with pytest.raises(ValueError):
        loader.load_adapter()
