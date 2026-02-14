from pyisolate._internal.serialization_registry import SerializerRegistry


def test_singleton_identity():
    r1 = SerializerRegistry.get_instance()
    r2 = SerializerRegistry.get_instance()
    assert r1 is r2


def test_register_and_lookup():
    registry = SerializerRegistry.get_instance()
    registry.clear()

    registry.register("Foo", lambda x: {"v": x}, lambda x: x["v"])

    assert registry.has_handler("Foo")
    serializer = registry.get_serializer("Foo")
    deserializer = registry.get_deserializer("Foo")

    payload = serializer(123) if serializer else None
    assert payload == {"v": 123}
    assert deserializer(payload) == 123 if deserializer else False


def test_clear_resets_handlers():
    registry = SerializerRegistry.get_instance()
    registry.register("Bar", lambda x: x)
    assert registry.has_handler("Bar")

    registry.clear()
    assert not registry.has_handler("Bar")
