import numpy as np
import math


def vector_divergence(grad_vec, dim):
    div = 0
    for d in range(dim):
        div += grad_vec[d][d]
    return div


def resistive_flux(eta, b, num_grad_dim, const_eta):
    # set up gradient test values
    grad_eta = np.empty(num_grad_dim)
    grad_b = np.empty((3, num_grad_dim))
    for dim in range(num_grad_dim):
        if (const_eta):
            grad_eta[dim] = 0.0
        else:
            grad_eta[dim] = eta * math.pow(-1, dim + 1) * (dim + 2)
        for i in range(3):
            grad_b[i][dim] = b[i] * math.pow(-1, dim) * (dim + 1)

    # first term: eta * grad_b
    eta_times_grad_b = eta * grad_b

    # second_term
    # div(eta B) = eta div(B) + grad(eta).B
    div_b = 0.0
    grad_eta_dot_b = 0.0
    for dim in range(num_grad_dim):
        div_b += grad_b[dim][dim]
        grad_eta_dot_b += grad_eta[dim] * b[dim]
    div_eta_b = np.zeros((3, num_grad_dim))
    for dim in range(num_grad_dim):
        div_eta_b[dim][dim] = eta * div_b + grad_eta_dot_b

    ind_flux = eta_times_grad_b - div_eta_b

    # third term B otimes grad(eta)
    # G_x = B_x d_eta/dx   G_y = B_y d_eta/dx   G_z = B_z d_eta/dx
    #       B_x d_eta/dy         B_y d_eta/dy         B_z d_eta/dy
    #       B_x d_eta/dz         B_y d_eta/dz         B_z d_eta/dz
    for i in range(num_grad_dim):
        for j in range(num_grad_dim):
            ind_flux[i][j] += b[j] * grad_eta[i]

    print("\ninduction eqn resistive flux ", num_grad_dim, "D:\n")
    print("\tind_0_flux = ", ind_flux[0])
    print("\tind_1_flux = ", ind_flux[1])
    print("\tind_2_flux = ", ind_flux[2])


# set the dependencies
eta = 0.16
b = np.array((1.1, 2.1, 3.1))
mu_0 = 0.05

hat_eta = eta / mu_0

# 2D test, variable resistivity
resistive_flux(hat_eta, b, 2, False)

# 3D test, variable resistivity
resistive_flux(hat_eta, b, 3, False)

# 2D test, constant resistivity
resistive_flux(hat_eta, b, 2, True)

# 3D test, constant resistivity
resistive_flux(hat_eta, b, 3, True)
