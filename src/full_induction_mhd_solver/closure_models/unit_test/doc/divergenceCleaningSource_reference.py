import numpy as np
import math


def get_exp_val_at_point(qp, num_grad_dim, num_space_dim):
    # set the dependencies
    max_dim = max(num_grad_dim, num_space_dim)
    grad_psi = np.zeros(max_dim)
    vel = np.zeros(max_dim)

    for d in range(num_grad_dim):
        grad_psi[d] = pow(-0.9, d + 1) * (qp + d + 1)
    for d in range(num_space_dim):
        vel[d] = pow(-0.6, d) * (qp + d + 1)

    exp_pot_src_val = -np.dot(vel, grad_psi)

    return exp_pot_src_val


def get_exp_values(num_grad_dim, num_space_dim):
    # get expected values at maximum of 8 quadrature points
    max_num_qp = 8
    exp_pot_src_vals = np.zeros(max_num_qp)
    for qp in range(max_num_qp):
        exp_pot_src_vals[qp] = get_exp_val_at_point(qp, num_grad_dim,
                                                    num_space_dim)

    print("Divergence Cleaning Source (num_grad_dim = ", num_grad_dim,
          ", num_space_dim = ", num_space_dim, ")")
    print("  exp_src_magn_pot[", max_num_qp, "] = ", exp_pot_src_vals, "\n")


get_exp_values(2, 2)
get_exp_values(2, 3)
get_exp_values(3, 3)
