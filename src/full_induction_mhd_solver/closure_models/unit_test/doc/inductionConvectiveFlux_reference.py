import numpy as np

# set the dependencies
v = np.array((1.25, 1.5, 1.125))
b = np.array((1.1, 2.1, 3.1))

psi = 0.4
pmag = 0.8
mu_0 = .05
c_h = 5.0

mtm_0_flux = np.array((0.125, -0.26, 0.377))
mtm_1_flux = np.array((0.250, -0.52, 0.754))
mtm_2_flux = np.array((0.375, -0.78, 1.131))

# compute the induction contribution to momentum flux
# rho_u fluxes in x, y, z directions
mtm_0_flux[0] += -b[0] * b[0] / mu_0 + pmag
mtm_0_flux[1] += -b[1] * b[0] / mu_0
mtm_0_flux[2] += -b[2] * b[0] / mu_0
# rho_v fluxes in x, y, z directions
mtm_1_flux[0] += -b[0] * b[1] / mu_0
mtm_1_flux[1] += -b[1] * b[1] / mu_0 + pmag
mtm_1_flux[2] += -b[2] * b[1] / mu_0
# r_ho_w fluxes in x, y, z directions
mtm_2_flux[0] += -b[0] * b[2] / mu_0
mtm_2_flux[1] += -b[1] * b[2] / mu_0
mtm_2_flux[2] += -b[2] * b[2] / mu_0 + pmag

# compute induction equation flux (without divergence cleaning)
# x direction flux for the induction equations [F_x(B_0), F_x(B_1), F_x(B_2)]
ind_flux_x = [
    v[0] * b[0] - b[0] * v[0], v[0] * b[1] - b[0] * v[1],
    v[0] * b[2] - b[0] * v[2]
]
# y direction flux for the induction equations [F_y(B_0), F_y(B_1), F_y(B_2)]
ind_flux_y = [
    v[1] * b[0] - b[1] * v[0], v[1] * b[1] - b[1] * v[1],
    v[1] * b[2] - b[1] * v[2]
]
# z direction flux for the induction equations [F_z(B_0), F_z(B_1), F_z(B_2)]
ind_flux_z = [
    v[2] * b[0] - b[2] * v[0], v[2] * b[1] - b[2] * v[1],
    v[2] * b[2] - b[2] * v[2]
]

# print expected momentum fluxes
print("\nExpected fluxes (no divergence cleaning)\n")
print("\tmtm_0_flux = ", mtm_0_flux)
print("\tmtm_1_flux = ", mtm_1_flux)
print("\tmtm_2_flux = ", mtm_2_flux)

# B_0 fluxes
print("\n\tind_0_flux = ", ind_flux_x[0], ind_flux_y[0], ind_flux_z[0])
# B_1 fluxes
print("\tind_1_flux = ", ind_flux_x[1], ind_flux_y[1], ind_flux_z[1])
# B_2 fluxes
print("\tind_2_flux = ", ind_flux_x[2], ind_flux_y[2], ind_flux_z[2])

# add divergence cleaning contributions
ind_flux_x[0] += c_h * psi
ind_flux_y[1] += c_h * psi
ind_flux_z[2] += c_h * psi
psi_flux = b * c_h

# print expected momentum fluxes
print("\nExpected fluxes (with divergence cleaning)\n")
print("\tmtm_0_flux = ", mtm_0_flux)
print("\tmtm_1_flux = ", mtm_1_flux)
print("\tmtm_2_flux = ", mtm_2_flux)

# B_0 fluxes
print("\n\tind_0_flux = ", ind_flux_x[0], ind_flux_y[0], ind_flux_z[0])
# B_1 fluxes
print("\tind_1_flux = ", ind_flux_x[1], ind_flux_y[1], ind_flux_z[1])
# B_2 fluxes
print("\tind_2_flux = ", ind_flux_x[2], ind_flux_y[2], ind_flux_z[2])

print("\n\tpsi_flux = ", psi_flux)
print()
