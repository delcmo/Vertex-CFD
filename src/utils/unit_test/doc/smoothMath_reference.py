import math
import numpy as np


# Returns normalized vector
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# The implementation of norm(x, tol) in SmoothMath
# for vectors
def smooth_norm(x, tol):
    dotp = np.dot(x, x)
    tol2 = tol * tol

    if (tol2 <= dotp):
        return np.sqrt(dotp)
    else:
        return 0.5 * (dotp + tol2) / tol


def deriv_smooth_norm(x, tol, dir):
    dotp = np.dot(x, x)
    tol2 = tol * tol

    if (tol2 <= dotp):
        return x[dir] / np.sqrt(dotp)
    else:
        return x[dir] / tol


# The implementation of norm(v, M, tol) in SmoothMath
# for vectors and a metric tensor
def smooth_metric_norm(x, M, tol):
    dotp = x.T @ M @ x
    tol2 = tol * tol

    if (tol2 <= dotp):
        return np.sqrt(dotp)
    else:
        return 0.5 * (dotp + tol2) / tol


# See the following
# https://math.stackexchange.com/questions/189434/derivative-of-quadratic-form
def deriv_smooth_metric_norm(x, M, tol, dir):
    dotp = x.T @ M @ x
    tol2 = tol * tol
    Dx = (M + M.T) @ x

    if (tol2 <= dotp):
        return 0.5 * Dx[dir] / np.sqrt(dotp)
    else:
        return 0.5 * Dx[dir] / tol


# Test vectors
v2 = np.array([0.25, -0.166])
v3 = np.array([0.175, 0.33, -0.92])

# Tolerance
tol = 1.0

# Print the norm of one of the vectors
print('smooth_norm v2: (0.0) ', smooth_norm(v2, 0.0))
print('smooth_norm v2: (%d) ' % tol, smooth_norm(v2, tol))

print('deriv_smooth_norm v2: (0.0) ', deriv_smooth_norm(v2, 0.0, 0))
print('deriv_smooth_norm v2: (0.0) ', deriv_smooth_norm(v2, 0.0, 1))

print('deriv_smooth_norm v2: (%d) ' % tol, deriv_smooth_norm(v2, tol, 0))
print('deriv_smooth_norm v2: (%d) ' % tol, deriv_smooth_norm(v2, tol, 1))

print('smooth_norm v3: (0.0) ', smooth_norm(v3, 0.0))
print('smooth_norm v3: (%d) ' % tol, smooth_norm(v3, tol))

print('deriv_smooth_norm v3: (0.0) ', deriv_smooth_norm(v3, 0.0, 0))
print('deriv_smooth_norm v3: (0.0) ', deriv_smooth_norm(v3, 0.0, 1))
print('deriv_smooth_norm v3: (0.0) ', deriv_smooth_norm(v3, 0.0, 2))

print('deriv_smooth_norm v3: (%d) ' % tol, deriv_smooth_norm(v3, tol, 0))
print('deriv_smooth_norm v3: (%d) ' % tol, deriv_smooth_norm(v3, tol, 1))
print('deriv_smooth_norm v3: (%d) ' % tol, deriv_smooth_norm(v3, tol, 2))

# Construct two orthonormal basis vectors
a = [1.0, 1.0]
b = [-1.0, 1.0]

# Normalize
a = normalize(a)
b = normalize(b)

# Build a matrix of eigenvectors
U = np.vstack([a, b])

# Element lengths
h1 = 0.5
h2 = 1.0

# Diagonal matrix
S = np.diag([h1, h2])

# Construct a 2d metric tensor
M2 = U.T @ S @ U
print(M2)
# Print the norm of one of the vectors
print('smooth_metric_norm v2: (0.0) ', smooth_metric_norm(v2, M2, 0.0))
print('smooth_metric_norm v2: (%d) ' % tol, smooth_metric_norm(v2, M2, tol))

print('deriv_smooth_metric_norm v2: (0.0) ',
      deriv_smooth_metric_norm(v2, M2, 0.0, 0))
print('deriv_smooth_metric_norm v2: (0.0) ',
      deriv_smooth_metric_norm(v2, M2, 0.0, 1))

print('deriv_smooth_metric_norm v2: (%d) ' % tol,
      deriv_smooth_metric_norm(v2, M2, tol, 0))
print('deriv_smooth_metric_norm v2: (%d) ' % tol,
      deriv_smooth_metric_norm(v2, M2, tol, 1))

# Now create a 3d orthonormal basis
a = [1.0, 1.0, 0]
b = [-1.0, 1.0, 0]
c = np.cross(a, b)
a = normalize(a)
b = normalize(b)
c = normalize(c)

U = np.vstack([a, b, c])

# Element lengths
h1 = 0.5
h2 = 0.25
h3 = 1.0

S = np.diag([h1, h2, h3])

# Construct a 3d metric tensor
M3 = U.T @ S @ U
print(M3)

print('smooth_metric_norm v3: (0.0) ', smooth_metric_norm(v3, M3, 0.0))
print('smooth_metric_norm v3: (%d) ' % tol, smooth_metric_norm(v3, M3, tol))

print('deriv_smooth_metric_norm v3: (0.0) ',
      deriv_smooth_metric_norm(v3, M3, 0.0, 0))
print('deriv_smooth_metric_norm v3: (0.0) ',
      deriv_smooth_metric_norm(v3, M3, 0.0, 1))
print('deriv_smooth_netric_norm v3: (0.0) ',
      deriv_smooth_metric_norm(v3, M3, 0.0, 2))

print('deriv_smooth_metric_norm v3: (%d) ' % tol,
      deriv_smooth_metric_norm(v3, M3, tol, 0))
print('deriv_smooth_metric_norm v3: (%d) ' % tol,
      deriv_smooth_metric_norm(v3, M3, tol, 1))
print('deriv_smooth_metric_norm v3: (%d) ' % tol,
      deriv_smooth_metric_norm(v3, M3, tol, 2))

# Now test the identity
M2 = np.array([[1.0, 0.0], [0.0, 1.0]])
print(M2)
print('smooth_metric_norm v2: (0.0) %.18f' % smooth_metric_norm(v2, M2, 0.0))
print('smooth_metric_norm v2: (%d) %.18f' %
      (tol, smooth_metric_norm(v2, M2, tol)))

print('deriv_smooth_metric_norm v2: (0.0) %.18f' %
      deriv_smooth_metric_norm(v2, M2, 0.0, 0))
print('deriv_smooth_metric_norm v2: (0.0) %.18f' %
      deriv_smooth_metric_norm(v2, M2, 0.0, 1))

print('deriv_smooth_metric_norm v2: (%d) %.18f' %
      (tol, deriv_smooth_metric_norm(v2, M2, tol, 0)))
print('deriv_smooth_metric_norm v2: (%d) %.18f' %
      (tol, deriv_smooth_metric_norm(v2, M2, tol, 1)))

M3 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
print(M3)
print('smooth_metric_norm v3: (0.0) %.18f' % smooth_metric_norm(v3, M3, 0.0))
print('smooth_metric_norm v3: (%d) %.18f' %
      (tol, smooth_metric_norm(v3, M3, tol)))

print('deriv_smooth_metric_norm v3: (0.0) %.18f' %
      deriv_smooth_metric_norm(v3, M3, 0.0, 0))
print('deriv_smooth_netric_norm v3: (0.0) %.18f' %
      deriv_smooth_metric_norm(v3, M3, 0.0, 1))
print('deriv_smooth_netric_norm v3: (0.0) %.18f' %
      deriv_smooth_metric_norm(v3, M3, 0.0, 2))

print('deriv_smooth_metric_norm v3: (%d) %.18f' %
      (tol, deriv_smooth_metric_norm(v3, M3, tol, 0)))
print('deriv_smooth_metric_norm v3: (%d) %.18f' %
      (tol, deriv_smooth_metric_norm(v3, M3, tol, 1)))
print('deriv_smooth_metric_norm v3: (%d) %.18f' %
      (tol, deriv_smooth_metric_norm(v3, M3, tol, 2)))
