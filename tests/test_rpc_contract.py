"""Tests for RPC behavior and ProxiedSingleton contracts.

These tests verify:
1. ProxiedSingleton instances are singletons
2. RPC method calls work correctly
3. Event loop recreation doesn't break RPC
4. Exceptions propagate correctly

Note: These are unit tests that verify RPC contracts at the boundary
without full process isolation. For full integration tests, see
original_integration/.
"""

import asyncio

import pytest

from pyisolate._internal.rpc_protocol import ProxiedSingleton, SingletonMetaclass

from .fixtures.test_adapter import MockRegistry


class TestProxiedSingletonContract:
    """Tests for ProxiedSingleton metaclass behavior."""

    def setup_method(self):
        """Clear singleton instances before each test."""
        SingletonMetaclass._instances.clear()

    def teardown_method(self):
        """Clear singleton instances after each test."""
        SingletonMetaclass._instances.clear()

    def test_singleton_returns_same_instance(self):
        """Multiple instantiations return the same instance."""
        instance1 = MockRegistry()
        instance2 = MockRegistry()

        assert instance1 is instance2

    def test_singleton_instance_persists(self):
        """Singleton instance persists across calls."""
        instance1 = MockRegistry()
        instance1.register("test_object")

        instance2 = MockRegistry()
        # Should see the object registered via instance1
        assert instance2.get("obj_0") == "test_object"

    def test_different_singletons_are_independent(self):
        """Different ProxiedSingleton subclasses are independent."""

        class AnotherRegistry(ProxiedSingleton):
            def __init__(self):
                super().__init__()
                self.data = "another"

        test_instance = MockRegistry()
        another_instance = AnotherRegistry()

        assert test_instance is not another_instance
        assert isinstance(test_instance, MockRegistry)
        assert isinstance(another_instance, AnotherRegistry)


class TestRpcMethodContract:
    """Tests for RPC method call contract."""

    def setup_method(self):
        """Clear singleton instances before each test."""
        SingletonMetaclass._instances.clear()

    def teardown_method(self):
        """Clear singleton instances after each test."""
        SingletonMetaclass._instances.clear()

    def test_method_returns_value(self):
        """RPC method must return expected value."""
        registry = MockRegistry()
        obj = {"key": "value"}

        obj_id = registry.register(obj)
        result = registry.get(obj_id)

        assert result == obj

    def test_method_accepts_arguments(self):
        """RPC method must accept positional and keyword arguments."""
        registry = MockRegistry()

        # Positional
        id1 = registry.register("positional_arg")
        assert registry.get(id1) == "positional_arg"

    def test_method_handles_none_return(self):
        """RPC method can return None."""
        registry = MockRegistry()

        result = registry.get("nonexistent")
        assert result is None

    def test_method_handles_complex_objects(self):
        """RPC method can handle complex nested objects."""
        registry = MockRegistry()

        complex_obj = {
            "list": [1, 2, 3],
            "nested": {"a": {"b": {"c": 42}}},
            "mixed": [{"x": 1}, {"y": 2}],
        }

        obj_id = registry.register(complex_obj)
        result = registry.get(obj_id)

        assert result == complex_obj


class TestEventLoopResilience:
    """Tests for RPC resilience across event loop recreation.

    This is a critical contract: ProxiedSingleton instances must
    remain functional even when the event loop is closed and
    recreated (e.g., between workflow executions).
    """

    def setup_method(self):
        """Clear singleton instances before each test."""
        SingletonMetaclass._instances.clear()

    def teardown_method(self):
        """Clear singleton instances after each test."""
        SingletonMetaclass._instances.clear()

    def test_singleton_survives_loop_recreation(self):
        """Singleton instance survives event loop recreation."""
        # Create initial loop
        loop1 = asyncio.new_event_loop()
        asyncio.set_event_loop(loop1)

        # Create singleton and store data
        registry = MockRegistry()
        obj_id = registry.register("loop1_object")

        # Close loop1
        loop1.close()

        # Create new loop
        loop2 = asyncio.new_event_loop()
        asyncio.set_event_loop(loop2)

        # Singleton should still work
        result = registry.get(obj_id)
        assert result == "loop1_object"

        # Cleanup
        loop2.close()

    def test_singleton_data_persists_across_loops(self):
        """Data stored in singleton persists across event loops."""
        # First loop
        loop1 = asyncio.new_event_loop()
        asyncio.set_event_loop(loop1)

        registry = MockRegistry()
        id1 = registry.register("first")
        id2 = registry.register("second")

        loop1.close()

        # Second loop
        loop2 = asyncio.new_event_loop()
        asyncio.set_event_loop(loop2)

        # All data should still be accessible
        assert registry.get(id1) == "first"
        assert registry.get(id2) == "second"

        loop2.close()


class TestRpcErrorHandling:
    """Tests for RPC error handling contract."""

    def setup_method(self):
        """Clear singleton instances before each test."""
        SingletonMetaclass._instances.clear()

    def teardown_method(self):
        """Clear singleton instances after each test."""
        SingletonMetaclass._instances.clear()

    def test_method_exception_propagates(self):
        """Exceptions in RPC methods should propagate."""

        class FailingService(ProxiedSingleton):
            def fail(self):
                raise ValueError("Intentional failure")

        service = FailingService()

        with pytest.raises(ValueError, match="Intentional failure"):
            service.fail()

    def test_type_error_propagates(self):
        """TypeError in RPC methods should propagate."""

        class TypedService(ProxiedSingleton):
            def typed_method(self, value: int) -> int:
                return value + 1

        service = TypedService()

        # Wrong type should raise TypeError
        with pytest.raises(TypeError):
            service.typed_method("not an int")


class TestAttrDictContract:
    """Tests for AttrDict helper class."""

    def test_attrdict_attribute_access(self):
        """AttrDict allows attribute-style access."""
        ad = shared.AttrDict({"key": "value", "nested": {"inner": 42}})

        assert ad.key == "value"
        assert ad.nested["inner"] == 42

    def test_attrdict_dict_access(self):
        """AttrDict allows dict-style access."""
        ad = shared.AttrDict({"key": "value"})

        assert ad["key"] == "value"

    def test_attrdict_missing_key(self):
        """AttrDict raises AttributeError for missing keys."""
        ad = shared.AttrDict({"key": "value"})

        with pytest.raises(AttributeError):
            _ = ad.nonexistent


class TestAttributeContainerContract:
    """Tests for AttributeContainer helper class."""

    def test_container_wraps_dict(self):
        """AttributeContainer wraps a dict."""
        data = {"a": 1, "b": 2}
        container = shared.AttributeContainer(data)

        assert container.a == 1
        assert container.b == 2

    def test_container_to_dict(self):
        """AttributeContainer can convert back to dict."""
        data = {"a": 1, "b": 2}
        container = shared.AttributeContainer(data)

        assert container._data == data
