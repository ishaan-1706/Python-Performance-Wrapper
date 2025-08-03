import numpy as np
from core.api import matmul

def test_matmul_identity():
    I = np.eye(3)
    M = np.array([[1,2,3],[4,5,6],[7,8,9]])
    # I @ M == M @ I == M
    out1 = matmul(I, M)
    out2 = matmul(M, I)
    assert isinstance(out1, np.ndarray)
    assert isinstance(out2, np.ndarray)
    assert np.allclose(out1, M)
    assert np.allclose(out2, M)

def test_matmul_small():
    A = np.array([[2,0],[1,3]])
    B = np.array([[1,4],[5,6]])
    expected = np.array([[2,8],[16,22]])
    assert np.array_equal(matmul(A, B), expected)