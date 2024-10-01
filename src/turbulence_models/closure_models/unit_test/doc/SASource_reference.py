import numpy as np
import math

# Thermophysical constants
nu = 0.25
wall_dist = 0.1
sa_vars = [-3.0, 3.0]

# SA model constants
sigma = 2.0 / 3.0
kappa = 0.41
cb1 = 0.1355
cb2 = 0.622
ct3 = 1.2
ct4 = 0.5
cv1 = 7.1
cv2 = 0.7
cv3 = 0.9
cw1 = cb1 / kappa / kappa + (1.0 + cb2) / sigma
cw2 = 0.3
cw3 = 2.0
rlim = 10

# 2D and 3D velocity gradients
grad_vel_2D = np.array([[-0.25, 0.5], [-0.5, 1.0]])
grad_vel_3D = np.array([[-0.25, 0.5, -0.75], [-0.5, 1.0, -1.5],
                        [-0.125, 0.25, -0.375]])

dims = [2, 3]

for dim in dims:
    for sa_var in sa_vars:
        print("Computing turbulence quantities in ", dim, "D")
        print(" with sa_var = ", sa_var)

        grad_vel = grad_vel_2D

        if (dim == 3):
            grad_vel = grad_vel_3D

        S2 = pow(grad_vel[1, 0] - grad_vel[0, 1], 2.0)

        if (dim == 3):
            S2 += (pow(grad_vel[2, 1] - grad_vel[1, 2], 2.0) +
                   pow(grad_vel[0, 2] - grad_vel[2, 0], 2.0))

        S = math.sqrt(S2)

        print("    S = ", S)

        grad_sa = 3 * grad_vel[0]

        sa_source = cb2 / sigma * np.dot(grad_sa, grad_sa)

        print("    grad sa contributuion: ", sa_source)

        chi = sa_var / nu

        fv1 = pow(chi, 3.0) / (pow(chi, 3.0) + pow(cv1, 3.0))
        fv2 = 1.0 - chi / (1 + chi * fv1)

        ft2 = ct3 * math.exp(-ct4 * chi * chi)

        Sbar = sa_var * fv2 / kappa / kappa / wall_dist / wall_dist

        Stilda = 0

        if (Sbar > -cv2 * S):
            Stilda = S + Sbar
        else:
            Stilda = S + S * (cv2 * cv2 * S + cv3 * Sbar) / (
                (cv3 - 2 * cv2) * S - Sbar)

        # Calculate production term
        sa_prod = 0.0
        if (sa_var > 0):
            sa_prod = cb1 * (1.0 - ft2) * Stilda * sa_var
        else:
            sa_prod = cb1 * (1.0 - ct3) * S * sa_var

        print("    sa_prod: ", sa_prod)

        # Calculations for destruction term
        r = min(sa_var / Stilda / kappa / kappa / wall_dist / wall_dist, rlim)
        g = r + cw2 * (pow(r, 6.0) - r)
        fw = g * pow(
            (1.0 + pow(cw3, 6.0)) / (pow(g, 6.0) + pow(cw3, 6.0)), 1.0 / 6.0)

        # Calculate destruction term
        sa_dest = 0.0
        if (sa_var > 0):
            sa_dest = -(cw1 * fw - cb1 * ft2 / kappa / kappa) * pow(
                sa_var / wall_dist, 2.0)
        else:
            sa_dest = cw1 * pow(sa_var / wall_dist, 2.0)
        print("    sa_dest: ", sa_dest)

        # Add production and destruction terms
        sa_source += (sa_prod + sa_dest)

        print("    total SA source: ", sa_source)
        print("")
