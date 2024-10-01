import numpy as np


def hartmann_pb_exact(M, L, y):
    return (np.cosh(M) - np.cosh(M * y / L)) / (np.cosh(M) - 1.0)


B = np.array([1.5, 2.0, 3.0])
B_magn = np.linalg.norm(B)
L = 2.5
sigma = 3.5
rho = 4.0
nu = 4.5
M = B_magn * L * np.sqrt(sigma / (rho * nu))

# output value
y = 0.5
print("Exact value:", hartmann_pb_exact(M, L, y))
