# tests/test_input_generator.py
import numpy as np
import pytest
from core.input_generator import generate_input

def test_generate_matmul():
    A, B = generate_input("matmul")
    assert A.shape[1] == B.shape[0]

def test_generate_monte_carlo():
    n = generate_input("monte_carlo")
    assert isinstance(n, int) and n > 0

def test_generate_gaussian_blur():
    img = generate_input("gaussian_blur")
    assert isinstance(img, list)
    assert len(img) == len(img[0])
