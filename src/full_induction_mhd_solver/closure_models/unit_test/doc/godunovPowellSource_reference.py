import numpy as np
import math


def get_exp_val_at_point(qp, num_space_dim):
    # set the dependencies
    mu_0 = 0.05
    div_mag = pow(-1.0, qp) * (qp + 1.1)
    vel = np.zeros(3)
    mag = np.zeros(3)
    for d in range(num_space_dim):
        vel[d] = pow(-0.6, d) * (qp + d + 1.2)
    for d in range(3):
        mag[d] = pow(-0.9, d + 1) * (qp + d + 1.3)

    exp_mtm_src_val = np.zeros(num_space_dim)
    exp_ind_src_val = np.zeros(num_space_dim)
    for d in range(num_space_dim):
        exp_mtm_src_val[d] = -div_mag * mag[d] / mu_0
        exp_ind_src_val[d] = -div_mag * vel[d]

    return (exp_mtm_src_val, exp_ind_src_val)


def get_exp_values(num_space_dim):
    # get expected values at maximum of 8 quadrature points
    max_num_qp = 8
    exp_mtm_src_vals = np.zeros((max_num_qp, num_space_dim))
    exp_ind_src_vals = np.zeros((max_num_qp, num_space_dim))
    for qp in range(max_num_qp):
        (exp_mtm_src_vals[qp],
         exp_ind_src_vals[qp]) = get_exp_val_at_point(qp, num_space_dim)

    print("Divergence Cleaning Source (num_space_dim = ", num_space_dim, ")")
    print("  exp_src_mtm[", max_num_qp, "][", num_space_dim, "] =\n",
          exp_mtm_src_vals, "\n")
    print("  exp_src_ind[", max_num_qp, "][", num_space_dim, "] =\n",
          exp_ind_src_vals, "\n")


get_exp_values(2)
get_exp_values(3)
