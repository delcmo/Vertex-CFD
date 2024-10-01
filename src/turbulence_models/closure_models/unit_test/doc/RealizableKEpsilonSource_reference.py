import numpy as np
import math

# Fluid properties
nu = 0.25

# Turbulent quantities
nu_t = 3.0
k = 4.0
e = 5.0

# Realizable KEpsilon model constants
C_2 = 1.9

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
    grad_u_sqr = 0.0

    for i in range(0, dim):
        for j in range(0, dim):
            grad_u_sqr += pow(grad_vel[i, j], 2.0)
            S2 += pow(0.5 * (grad_vel[i, j] + grad_vel[j, i]), 2.0)

    S = math.sqrt(2 * S2)

    print("    S = ", S, "\n")
    print("    grad_u_sqr = ", grad_u_sqr, "\n")

    eta = S * k / e
    C_1 = max(0.43, eta / (5 + eta))

    print("    eta = ", eta, "\n")
    print("    C_1 = ", C_1, "\n")

    k_prod = nu_t * grad_u_sqr
    k_dest = -e
    k_source = k_prod + k_dest

    print("    k prod: ", k_prod)
    print("    k dest: ", k_dest)
    print("    k source: ", k_source, "\n")

    e_prod = C_1 * S * e
    e_dest = -C_2 * pow(e, 2.0) / (k + math.sqrt(nu * e))
    e_source = e_prod + e_dest

    print("    e prod: ", e_prod)
    print("    e dest: ", e_dest)
    print("    e source: ", e_source, "\n")
