"""Unit tests for SingletonMetaclass and ProxiedSingleton behavior."""


import pytest

from pyisolate._internal.shared import (
    LocalMethodRegistry,
    ProxiedSingleton,
    SingletonMetaclass,
    local_execution,
)


@pytest.fixture(autouse=True)
def reset_singleton_state():
    """Ensure singleton/global registries are clean for every test."""
    SingletonMetaclass._instances.clear()
    LocalMethodRegistry._instance = None
    yield
    SingletonMetaclass._instances.clear()
    LocalMethodRegistry._instance = None


class FakeCaller:
    """Minimal callable returned by FakeRPC.create_caller."""

    def __init__(self, target_cls, object_id):
        self.target_cls = target_cls
        self.object_id = object_id


class FakeRPC:
    """Capture create_caller invocations without spinning up real RPC."""

    def __init__(self):
        self.calls = []

    def create_caller(self, cls, object_id):
        caller = FakeCaller(cls, object_id)
        self.calls.append((cls, object_id, caller))
        return caller


class BasicSingleton(ProxiedSingleton):
    async def ping(self):  # pragma: no cover - method invoked via proxy
        return "pong"


class LocalMethodSingleton(ProxiedSingleton):
    def __init__(self):
        super().__init__()
        self.counter = 0

    @local_execution
    def increment(self):
        self.counter += 1
        return self.counter


class ChildSingleton(ProxiedSingleton):
    async def child_call(self):  # pragma: no cover
        return "child"


class ParentSingleton(ProxiedSingleton):
    child: ChildSingleton

    async def parent_call(self):  # pragma: no cover
        return "parent"


class TestSingletonMetaclass:
    def test_inject_instance_after_instantiation_raises(self):
        """inject_instance must run before first instantiation."""
        BasicSingleton()
        with pytest.raises(AssertionError):
            SingletonMetaclass.inject_instance(BasicSingleton, object())

    def test_get_remote_id_defaults_to_class_name(self):
        assert BasicSingleton.get_remote_id() == "BasicSingleton"


class TestUseRemote:
    def test_use_remote_sets_proxy_instance(self):
        """use_remote should inject proxy returned by RPC."""
        rpc = FakeRPC()
        BasicSingleton.use_remote(rpc)

        assert BasicSingleton in SingletonMetaclass._instances
        proxy = SingletonMetaclass._instances[BasicSingleton]
        assert isinstance(proxy, FakeCaller)
        assert proxy.target_cls is BasicSingleton
        assert rpc.calls[0][1] == BasicSingleton.get_remote_id()

    def test_local_execution_methods_registered(self):
        """Classes with @local_execution should be tracked by registry."""
        rpc = FakeRPC()
        LocalMethodSingleton.use_remote(rpc)

        registry = LocalMethodRegistry.get_instance()
        assert registry.is_local_method(LocalMethodSingleton, "increment")

        local_impl = registry.get_local_method(LocalMethodSingleton, "increment")
        assert local_impl() == 1
        assert local_impl() == 2  # local state should be preserved per process

    def test_nested_singletons_receive_callers(self):
        """Type-hinted ProxiedSingleton attributes get caller proxies injected."""
        rpc = FakeRPC()
        ParentSingleton.use_remote(rpc)

        parent_proxy = SingletonMetaclass._instances[ParentSingleton]
        assert isinstance(parent_proxy, FakeCaller)

        # The first call registers parent, the second should register child attribute
        assert len(rpc.calls) == 2
        # rpc.calls[-1] corresponds to child proxy creation
        _, child_object_id, child_proxy = rpc.calls[-1]
        assert child_object_id == ChildSingleton.get_remote_id()
        assert isinstance(child_proxy, FakeCaller)

        # Attribute on remote should reference the same child proxy
        assert parent_proxy.child is child_proxy


class TestLocalMethodRegistry:
    def test_get_local_method_requires_registration(self):
        """Attempting to access unregistered class should raise."""
        registry = LocalMethodRegistry.get_instance()
        with pytest.raises(ValueError):
            registry.get_local_method(BasicSingleton, "ping")

    def test_register_class_initializes_local_instance(self):
        registry = LocalMethodRegistry.get_instance()
        registry.register_class(LocalMethodSingleton)
        local_impl = registry.get_local_method(LocalMethodSingleton, "increment")
        assert callable(local_impl)
        assert local_impl() == 1
