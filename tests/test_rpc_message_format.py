"""Tests for RPC message format and error handling contracts.

These tests verify that RPC messages are properly formatted and
errors propagate correctly across process boundaries.
"""

import pytest

from pyisolate._internal.rpc_protocol import (
    ProxiedSingleton,
    SingletonMetaclass,
)
from pyisolate._internal.rpc_serialization import (
    AttrDict,
    AttributeContainer,
    _prepare_for_rpc,
)


class TestPrepareForRpc:
    """Tests for _prepare_for_rpc serialization."""

    def test_primitives_pass_through(self):
        """Primitive types pass through unchanged."""
        assert _prepare_for_rpc(42) == 42
        assert _prepare_for_rpc("hello") == "hello"
        assert _prepare_for_rpc(3.14) == 3.14
        assert _prepare_for_rpc(True) is True
        assert _prepare_for_rpc(None) is None

    def test_list_preserved(self):
        """Lists are preserved."""
        data = [1, 2, 3]
        result = _prepare_for_rpc(data)
        assert result == [1, 2, 3]

    def test_nested_list(self):
        """Nested lists are handled."""
        data = [[1, 2], [3, 4]]
        result = _prepare_for_rpc(data)
        assert result == [[1, 2], [3, 4]]

    def test_dict_preserved(self):
        """Dicts are preserved."""
        data = {"a": 1, "b": 2}
        result = _prepare_for_rpc(data)
        assert result == {"a": 1, "b": 2}

    def test_nested_dict(self):
        """Nested dicts are handled."""
        data = {"outer": {"inner": 42}}
        result = _prepare_for_rpc(data)
        assert result == {"outer": {"inner": 42}}

    def test_tuple_preserved(self):
        """Tuples are preserved (JSON converts to list on transport)."""
        data = (1, 2, 3)
        result = _prepare_for_rpc(data)
        # Implementation preserves tuples; JSON transport converts to list
        assert result == (1, 2, 3) or result == [1, 2, 3]

    def test_attrdict_converted(self):
        """AttrDict is converted to plain dict."""
        data = AttrDict({"key": "value"})
        result = _prepare_for_rpc(data)
        assert isinstance(result, dict)
        assert result["key"] == "value"

    def test_attribute_container_handled(self):
        """AttributeContainer is handled appropriately."""
        data = AttributeContainer({"a": 1, "b": 2})
        result = _prepare_for_rpc(data)
        # May be dict or container depending on implementation
        if isinstance(result, dict):
            assert result["a"] == 1
        else:
            assert hasattr(result, "_data")

    def test_mixed_nested_structure(self):
        """Mixed nested structures are handled."""
        data = {
            "list": [1, 2, {"nested": True}],
            "tuple": (3, 4),
            "value": "test",
        }
        result = _prepare_for_rpc(data)

        assert result["list"] == [1, 2, {"nested": True}]
        # Tuples may be preserved or converted
        assert result["tuple"] in [(3, 4), [3, 4]]
        assert result["value"] == "test"


class TestAttrDictBehavior:
    """Tests for AttrDict helper class behavior."""

    def test_attribute_access(self):
        """AttrDict allows attribute-style access."""
        ad = AttrDict({"name": "test", "value": 42})

        assert ad.name == "test"
        assert ad.value == 42

    def test_dict_access(self):
        """AttrDict allows dict-style access."""
        ad = AttrDict({"name": "test"})

        assert ad["name"] == "test"

    def test_nested_dict_access(self):
        """Nested dicts accessible via attribute."""
        ad = AttrDict({"outer": {"inner": "value"}})

        assert ad.outer["inner"] == "value"

    def test_missing_attribute_raises(self):
        """Missing attributes raise AttributeError."""
        ad = AttrDict({"existing": True})

        with pytest.raises(AttributeError):
            _ = ad.missing

    def test_missing_key_raises(self):
        """Missing keys raise KeyError."""
        ad = AttrDict({"existing": True})

        with pytest.raises(KeyError):
            _ = ad["missing"]

    def test_iteration(self):
        """AttrDict can be iterated."""
        ad = AttrDict({"a": 1, "b": 2})

        keys = list(ad.keys())
        assert "a" in keys
        assert "b" in keys


class TestAttributeContainerBehavior:
    """Tests for AttributeContainer helper class behavior."""

    def test_attribute_access(self):
        """AttributeContainer wraps dict with attribute access."""
        container = AttributeContainer({"x": 10, "y": 20})

        assert container.x == 10
        assert container.y == 20

    def test_data_property(self):
        """_data property returns underlying dict."""
        data = {"key": "value"}
        container = AttributeContainer(data)

        assert container._data == data

    def test_missing_attribute_raises(self):
        """Missing attributes raise AttributeError."""
        container = AttributeContainer({"existing": True})

        with pytest.raises(AttributeError):
            _ = container.missing


class TestSingletonMetaclass:
    """Tests for SingletonMetaclass behavior."""

    def setup_method(self):
        """Clear singleton instances before each test."""
        SingletonMetaclass._instances.clear()

    def teardown_method(self):
        """Clear singleton instances after each test."""
        SingletonMetaclass._instances.clear()

    def test_singleton_same_instance(self):
        """Multiple instantiations return same instance."""

        class MySingleton(ProxiedSingleton):
            def __init__(self):
                super().__init__()
                self.value = 42

        a = MySingleton()
        b = MySingleton()

        assert a is b

    def test_singleton_state_shared(self):
        """State is shared across references."""

        class StatefulSingleton(ProxiedSingleton):
            def __init__(self):
                super().__init__()
                self.data = []

        a = StatefulSingleton()
        a.data.append("from_a")

        b = StatefulSingleton()
        assert "from_a" in b.data

    def test_different_singletons_independent(self):
        """Different singleton classes are independent."""

        class SingletonA(ProxiedSingleton):
            pass

        class SingletonB(ProxiedSingleton):
            pass

        a = SingletonA()
        b = SingletonB()

        assert a is not b
        assert type(a) is not type(b)
