import numpy as np

# Coefficients
nu = 0.375
rho = 1.0

# 2-D case
grad_vel = np.array([[-0.25, 0.5], [-0.5, 1.0]])
print("\n2-D case:")

strain_rate_2D = 0.5 * (grad_vel + grad_vel.transpose())

fv_viscous_heat_2D = 2.0 * rho * nu * np.tensordot(strain_rate_2D,
                                                   strain_rate_2D)

print("fv_viscous_heat:", fv_viscous_heat_2D)

# 3-D case
grad_vel = np.array([[-0.25, 0.5, -0.75], [-0.5, 1.0, -1.5],
                     [-0.125, 0.25, -0.375]])

strain_rate = 0.5 * (grad_vel + grad_vel.transpose())

print("\n3-D case:\n")

fv_viscous_heat = 2.0 * rho * nu * np.tensordot(strain_rate, strain_rate)

print("fv_viscous_heat:", fv_viscous_heat)
