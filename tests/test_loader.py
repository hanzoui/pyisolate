
import pytest

from pyisolate._internal import loader


class DummyAdapter:
    identifier = "dummy"

    def __init__(self):
        self.instantiated = True


class DummyEntryPoint:
    def __init__(self, name, value):
        self.name = name
        self._value = value

    def load(self):
        return self._value


class EntryPointsObj:
    def __init__(self, eps):
        self._eps = eps

    def select(self, **kwargs):
        return self._eps


@pytest.fixture(autouse=True)
def reset_env(monkeypatch):
    monkeypatch.delenv("PYISOLATE_ADAPTER_OVERRIDE", raising=False)
    yield


def _patch_entry_points(monkeypatch, eps_list):
    obj = EntryPointsObj(eps_list)
    monkeypatch.setattr(loader, "entry_points", None, raising=False)

    def fake_entry_points():
        return obj

    # Patch both importlib.metadata and importlib_metadata to be safe
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    monkeypatch.setattr(loader, "entry_points", fake_entry_points, raising=False)

    try:
        import importlib_metadata  # type: ignore

        monkeypatch.setattr(importlib_metadata, "entry_points", fake_entry_points)
    except ImportError:
        pass  # importlib_metadata not installed, fine for Python 3.10+


def test_load_adapter_by_name(monkeypatch):
    eps = [DummyEntryPoint("dummy", DummyAdapter)]
    _patch_entry_points(monkeypatch, eps)

    adapter = loader.load_adapter("dummy")
    assert isinstance(adapter, DummyAdapter)


def test_auto_detect_single(monkeypatch):
    eps = [DummyEntryPoint("solo", DummyAdapter)]
    _patch_entry_points(monkeypatch, eps)

    adapter = loader.load_adapter()
    assert adapter.identifier == "dummy" or adapter.identifier == "solo"


def test_missing_adapter_raises(monkeypatch):
    eps = [DummyEntryPoint("other", DummyAdapter)]
    _patch_entry_points(monkeypatch, eps)

    with pytest.raises(ValueError):
        loader.load_adapter("missing")


def test_multiple_adapters_without_name(monkeypatch):
    eps = [DummyEntryPoint("a", DummyAdapter), DummyEntryPoint("b", DummyAdapter)]
    _patch_entry_points(monkeypatch, eps)

    with pytest.raises(ValueError):
        loader.load_adapter()


def test_env_override(monkeypatch):
    eps = [DummyEntryPoint("first", DummyAdapter), DummyEntryPoint("second", DummyAdapter)]
    _patch_entry_points(monkeypatch, eps)
    monkeypatch.setenv("PYISOLATE_ADAPTER_OVERRIDE", "second")

    adapter = loader.load_adapter("first")
    assert isinstance(adapter, DummyAdapter)
    # Override should take precedence even when name is provided
    assert adapter.identifier == "dummy"
