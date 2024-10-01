import numpy as np
import math

# Realizable K-Epsilon model constants
C_w = 0.500
C_k = 0.094

# 2D and 3D velocity gradients
grad_vel_2D = np.array([[-0.25, 0.5], [-0.5, 1.0]])
grad_vel_3D = np.array([[-0.25, 0.5, -0.75], [-0.5, 1.0, -1.5],
                        [-0.125, 0.25, -0.375]])

dim_list = [2, 3]

for dim in dim_list:
    print("Computing turbulence quantities in ", dim, "D\n")

    grad_vel = grad_vel_2D

    if (dim == 3):
        grad_vel = grad_vel_3D

    # Element length (=3x du/dx_i)
    element_length = 3.0 * grad_vel[0, :]
    delta = np.linalg.norm(element_length)

    # Find symmetric and antisymmetric velocity gradient tensors
    S = 0.5 * (grad_vel + np.transpose(grad_vel))
    W = 0.5 * (grad_vel - np.transpose(grad_vel))

    # Find magnitude squared of S and W
    magSqrS = pow(np.linalg.norm(S), 2.0)
    magSqrW = pow(np.linalg.norm(W), 2.0)

    # Construct S_d tensor
    S_d = np.matmul(S, S) + np.matmul(
        W, W) - 1.0 / 3.0 * np.identity(dim) * (magSqrS - magSqrW)
    magSqrSd = pow(np.linalg.norm(S_d), 2.0)

    # Compute sub-grid eddy viscosity
    nu_sgs = pow(C_w * delta, 2.0) * pow(magSqrSd, 3.0 / 2.0) / (
        pow(magSqrS, 5.0 / 2.0) + pow(magSqrSd, 5.0 / 4.0))

    # Compute sub-grid kinetic energy
    k_sgs = pow(nu_sgs / C_k / delta, 2.0)

    # Output information
    print("    magSqrS = ", magSqrS, "\n")
    print("    magSqrW = ", magSqrW, "\n")
    print("    magSqrSd = ", magSqrSd, "\n")
    print("    delta = ", delta, "\n")
    print("    nu_sgs = ", nu_sgs, "\n")
    print("    k_sgs = ", k_sgs, "\n\n")
