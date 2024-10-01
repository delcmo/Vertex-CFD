import numpy as np

# Coefficients
nu = 0.375
rho = 1.5

# 2-D case
print("\n2-D case:")
grad_vel = np.array([[-0.25, 0.5], [-0.5, 1.0]])
normals = np.array([-0.125, 0.25])

tau_dot_n = grad_vel.dot(normals)
tau_dot_n = np.linalg.norm(tau_dot_n)

print("Shear stress:", rho * nu * tau_dot_n)
print("Friction velocity:", np.sqrt(tau_dot_n * nu))

# 3-D case
print("\n3-D case:")
grad_vel = np.array([[-0.25, 0.5, -0.75], [-0.5, 1.0, -1.5],
                     [-0.725, 1.45, -2.175]])
normals = np.array([-0.125, 0.25, -0.375])

tau_dot_n = grad_vel.dot(normals)
tau_dot_n = np.linalg.norm(tau_dot_n)

print("Shear stress:", rho * nu * tau_dot_n)
print("Friction velocity:", np.sqrt(tau_dot_n * nu), "\n")
strain_rate = 0.5 * (grad_vel + grad_vel.transpose())
