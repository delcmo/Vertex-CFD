import numpy as np

# Dependent variables
sigma = 3.0
B = np.array([1.1, 2.0, -0.3])
vel = np.array([0.1, -0.2, 0.3])


# Function to compute Lorentz force
def lorentz_force(sigma, grad_phi, B, dim):
    lf = -np.cross(grad_phi, B)[0:dim]
    print(lf)
    lf += np.dot(B[0:dim], vel[0:dim]) * B[0:dim]
    lf -= np.dot(B[0:dim], B[0:dim]) * vel[0:dim]
    lf *= sigma
    return lf


# 2D
grad_phi = np.array([0.6, 0.4, 0.0])
print("2-D Florentz:", lorentz_force(sigma, grad_phi, B, 2))

# 3D
grad_phi = np.array([0.6, 0.4, 0.5])
print("3-D Florentz:", lorentz_force(sigma, grad_phi, B, 3))
