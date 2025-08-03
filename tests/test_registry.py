# tests/test_registry.py
import pytest
from core.registry import list_ops, get_impls, register_impl
from core.types import ImplementationType

def dummy_op(x):
    return x

@pytest.fixture(autouse=True)
def cleanup_registry(monkeypatch):
    # isolate registry for each test
    from core import registry
    monkeypatch.setattr(registry, "_registry", {})
    yield

def test_register_and_list_ops():
    from core.registry import register_impl
    @register_impl("foo", "bar", ImplementationType.BASELINE)
    def foo_bar(x): return x
    assert "foo" in list_ops()
    impls = get_impls("foo")
    assert impls[0][0] == "bar"

