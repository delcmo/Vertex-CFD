# Import Python module to caculcate the singular values
import numpy
from numpy import array
from scipy.linalg import svd
from decimal import *


# Function to compute the singular values of the matrix A.
# It prints the eigenvalues and starts with the maximum eigenvalue
def compute_singular_values(A):
    U, s, VT = svd(A)
    for l in s:
        print("Singular values: %.17f" % l)


# qp = 0
print("\033[1mCompute singular values for qp = 0:\033[0m")
J = array([[4.0, 0.0], [0.0, 9.0]])
compute_singular_values(J)

# qp = 1
print("\n\033[1mCompute singular values for qp = 1:\033[0m")
J = array([[2.0, 6.0], [6.0, 2.0]])
compute_singular_values(J)

# qp = 2
print("\n\033[1mCompute singular values for qp = 2:\033[0m")
J = array([[-0.2, 0.3], [0.4, 0.5]])
compute_singular_values(J)

# qp = 3
print("\n\033[1mCompute singular values for qp = 3:\033[0m")
J = array([[2.0, -2.0], [-2.0, 2.0]])
compute_singular_values(J)
