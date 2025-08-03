# smoke_test.py

from multiprocessing import freeze_support

def main():
    import numpy as np
    from core.api import matmul, monte_carlo, gaussian_blur

    # 1. Matrix multiply
    A = np.random.rand(100, 100)
    B = np.random.rand(100, 100)
    C = matmul(A, B)
    print("matmul ->", C.shape)

    # 2. Monte Carlo π estimate
    pi_est = monte_carlo(100_000, seed=42)
    print("monte_carlo ->", pi_est)

    # 3. Gaussian blur
    img = np.random.rand(50, 50)
    blurred = gaussian_blur(img, sigma=1.0)
    print("gaussian_blur ->", blurred.shape)

if __name__ == "__main__":
    freeze_support()
    main()