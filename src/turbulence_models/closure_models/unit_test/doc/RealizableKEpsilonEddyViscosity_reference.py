import numpy as np
import math

# Turbulent quantities
k = 4.0
e = 5.0

# Realizable K-Epsilon model constants
A_0 = 4.0

# 2D and 3D velocity gradients
grad_vel_2D = np.array([[-0.25, 0.5], [-0.5, 1.0]])
grad_vel_3D = np.array([[-0.25, 0.5, -0.75], [-0.5, 1.0, -1.5],
                        [-0.125, 0.25, -0.375]])

dims = [2, 3]

for dim in dims:
    print("Computing turbulence quantities in ", dim, "D\n")

    grad_vel = grad_vel_2D

    if (dim == 3):
        grad_vel = grad_vel_3D

    S2 = 0.0
    Omega2 = 0.0
    W = 0.0

    # COMMENT: The Omega2 calculation must change if the equations
    # are to be solved in a rotating reference frame
    for i in range(0, dim):
        for j in range(0, dim):
            S2 += pow(0.5 * (grad_vel[i, j] + grad_vel[j, i]), 2.0)
            Omega2 += pow(0.5 * (grad_vel[i, j] - grad_vel[j, i]), 2.0)
            for l in range(0, dim):
                W += 1.0 / 8.0 * (grad_vel[i, j] + grad_vel[j, i]) * (
                    grad_vel[j, l] + grad_vel[l, j]) * (grad_vel[l, i] +
                                                        grad_vel[i, l])

    W = W / pow(S2, 3.0 / 2.0)

    print("    S2 = ", S2, "\n")
    print("    Omega2 = ", Omega2, "\n")
    print("    W = ", W, "\n")

    Us = math.sqrt(S2 + Omega2)
    phi = 1.0 / 3.0 * math.acos(max(min(math.sqrt(6.0) * W, 1.0), -1.0))
    As = math.sqrt(6.0) * math.cos(phi)

    C_nu = 1.0 / (A_0 + (As * Us * k / e))

    print("    Us = ", Us, "\n")
    print("    phi = ", phi, "\n")
    print("    As = ", As, "\n")
    print("    C_nu = ", C_nu, "\n")

    nu_t = C_nu * k * k / e

    print("    nu_t = ", nu_t, "\n\n")
