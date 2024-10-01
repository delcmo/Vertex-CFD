import numpy as np

# electrical conductivity
sigma = 6.25

# 2D
B = np.array([0.0, 0.0, 0.3])
v = np.array([0.5, 1.5, 0.0])

flux = sigma * np.cross(v, B)
print("2-D flux:", flux)

# 3D
B = np.array([1.1, 2.0, -0.3])
v = np.array([0.5, 1.5, 2.5])

flux = sigma * np.cross(v, B)
print("3-D flux:", flux)
