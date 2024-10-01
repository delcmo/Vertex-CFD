import numpy as np

# electrical conductivity
sigma = 6.25

# 2D
grad_phi = np.array([0.1, -0.2, 0.0])
B = np.array([0.0, 0.0, 0.3])
v = np.array([0.5, 1.5, 0.0])

J = sigma * (np.cross(v, B) - grad_phi)
print("2-D J:", J)

# 3D
grad_phi = np.array([0.1, -0.2, 0.3])
B = np.array([1.1, 2.0, -0.3])
v = np.array([0.5, 1.5, 2.5])

J = sigma * (np.cross(v, B) - grad_phi)
print("3-D J:", J)
