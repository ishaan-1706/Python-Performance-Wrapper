import pytest
from core.api import monte_carlo

def test_monte_carlo_return_type():
    pi = monte_carlo(10_000, seed=42)
    assert isinstance(pi, float)

def test_monte_carlo_reproducible():
    a = monte_carlo(50_000, seed=123)
    b = monte_carlo(50_000, seed=123)
    assert pytest.approx(a, rel=1e-6) == b