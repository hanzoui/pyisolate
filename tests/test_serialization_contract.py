"""Tests for SerializerRegistry and type serialization contracts.

These tests verify:
1. SerializerRegistry can register custom serializers
2. Serializers produce JSON-compatible output
3. Deserializers reconstruct objects correctly
4. Roundtrip serialization preserves data

Note: These are unit tests that verify serialization at the boundary.
They use the MockHostAdapter's serializers as the reference implementation.
"""

import json

from pyisolate._internal.serialization_registry import SerializerRegistry

from .fixtures.test_adapter import MockHostAdapter, MockTestData


class TestSerializerRegistryContract:
    """Tests for SerializerRegistry protocol compliance."""

    def setup_method(self):
        """Get fresh registry for each test."""
        self.registry = SerializerRegistry.get_instance()
        self.registry.clear()

    def teardown_method(self):
        """Clear registry after each test."""
        self.registry.clear()

    def test_registry_is_singleton(self):
        """SerializerRegistry.get_instance() returns same instance."""
        reg1 = SerializerRegistry.get_instance()
        reg2 = SerializerRegistry.get_instance()

        assert reg1 is reg2

    def test_register_serializer(self):
        """Can register a serializer for a type."""
        def serialize(obj):
            return {"value": obj}

        self.registry.register("MyType", serialize)

        assert self.registry.has_handler("MyType")

    def test_register_with_deserializer(self):
        """Can register both serializer and deserializer."""
        def serialize(obj):
            return {"v": obj}

        def deserialize(data):
            return data["v"]

        self.registry.register("MyType", serialize, deserialize)

        assert self.registry.get_serializer("MyType") is not None
        assert self.registry.get_deserializer("MyType") is not None

    def test_get_serializer_returns_callable(self):
        """get_serializer returns the registered callable."""
        def my_serializer(obj):
            return str(obj)

        self.registry.register("StringType", my_serializer)

        retrieved = self.registry.get_serializer("StringType")
        assert retrieved is my_serializer

    def test_get_deserializer_returns_callable(self):
        """get_deserializer returns the registered callable."""
        def my_deserializer(data):
            return int(data)

        self.registry.register("IntType", lambda x: x, my_deserializer)

        retrieved = self.registry.get_deserializer("IntType")
        assert retrieved is my_deserializer

    def test_get_unregistered_returns_none(self):
        """get_serializer/get_deserializer return None for unknown types."""
        assert self.registry.get_serializer("UnknownType") is None
        assert self.registry.get_deserializer("UnknownType") is None

    def test_has_handler_false_for_unknown(self):
        """has_handler returns False for unregistered types."""
        assert self.registry.has_handler("UnknownType") is False

    def test_has_handler_true_for_registered(self):
        """has_handler returns True for registered types."""
        self.registry.register("KnownType", lambda x: x)

        assert self.registry.has_handler("KnownType") is True


class TestSerializationRoundtrip:
    """Tests for serialization roundtrip correctness."""

    def setup_method(self):
        """Set up registry with MockHostAdapter serializers."""
        self.registry = SerializerRegistry.get_instance()
        self.registry.clear()

        adapter = MockHostAdapter()
        adapter.register_serializers(self.registry)

    def teardown_method(self):
        """Clear registry after each test."""
        self.registry.clear()

    def test_testdata_roundtrip(self):
        """TestData survives serialization roundtrip."""
        original = MockTestData("hello world")

        serializer = self.registry.get_serializer("MockTestData")
        deserializer = self.registry.get_deserializer("MockTestData")

        serialized = serializer(original)
        deserialized = deserializer(serialized)

        assert deserialized == original

    def test_testdata_with_int_value(self):
        """TestData with int value roundtrips correctly."""
        original = MockTestData(42)

        serializer = self.registry.get_serializer("MockTestData")
        deserializer = self.registry.get_deserializer("MockTestData")

        serialized = serializer(original)
        deserialized = deserializer(serialized)

        assert deserialized.value == 42

    def test_testdata_with_list_value(self):
        """TestData with list value roundtrips correctly."""
        original = MockTestData([1, 2, 3])

        serializer = self.registry.get_serializer("MockTestData")
        deserializer = self.registry.get_deserializer("MockTestData")

        serialized = serializer(original)
        deserialized = deserializer(serialized)

        assert deserialized.value == [1, 2, 3]

    def test_testdata_with_dict_value(self):
        """TestData with dict value roundtrips correctly."""
        original = MockTestData({"key": "value", "nested": {"a": 1}})

        serializer = self.registry.get_serializer("MockTestData")
        deserializer = self.registry.get_deserializer("MockTestData")

        serialized = serializer(original)
        deserialized = deserializer(serialized)

        assert deserialized.value == {"key": "value", "nested": {"a": 1}}


class TestJsonCompatibility:
    """Tests that serialized output is JSON-compatible."""

    def setup_method(self):
        """Set up registry with MockHostAdapter serializers."""
        self.registry = SerializerRegistry.get_instance()
        self.registry.clear()

        adapter = MockHostAdapter()
        adapter.register_serializers(self.registry)

    def teardown_method(self):
        """Clear registry after each test."""
        self.registry.clear()

    def test_serialized_is_json_serializable(self):
        """Serialized output must be JSON-serializable."""
        original = MockTestData("test")

        serializer = self.registry.get_serializer("MockTestData")
        serialized = serializer(original)

        # Must not raise
        json_str = json.dumps(serialized)
        assert json_str

    def test_json_roundtrip_preserves_data(self):
        """Data survives JSON serialization between processes."""
        original = MockTestData({"complex": [1, 2, {"nested": True}]})

        serializer = self.registry.get_serializer("MockTestData")
        deserializer = self.registry.get_deserializer("MockTestData")

        # Simulate cross-process: serialize -> JSON -> deserialize
        serialized = serializer(original)
        json_str = json.dumps(serialized)
        parsed = json.loads(json_str)
        deserialized = deserializer(parsed)

        assert deserialized == original


class TestSerializerProtocolCompliance:
    """Tests for SerializerRegistryProtocol compliance."""

    def test_registry_matches_protocol(self):
        """SerializerRegistry implements SerializerRegistryProtocol."""

        registry = SerializerRegistry.get_instance()

        # Check protocol methods exist
        assert hasattr(registry, "register")
        assert hasattr(registry, "get_serializer")
        assert hasattr(registry, "get_deserializer")
        assert hasattr(registry, "has_handler")

        # Check methods are callable
        assert callable(registry.register)
        assert callable(registry.get_serializer)
        assert callable(registry.get_deserializer)
        assert callable(registry.has_handler)

    def test_register_signature(self):
        """register() accepts type_name, serializer, and optional deserializer."""
        registry = SerializerRegistry.get_instance()
        registry.clear()

        # With deserializer
        registry.register("Type1", lambda x: x, lambda x: x)

        # Without deserializer
        registry.register("Type2", lambda x: x)

        assert registry.has_handler("Type1")
        assert registry.has_handler("Type2")
