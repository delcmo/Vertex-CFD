# Functions
def ic2d(y, h_min, h_max, vel_avg):
    coeff = 3.0 / 2.0
    H = 0.5 * (h_max - h_min)
    u = vel_avg * coeff * (1.0 - y * y / (H * H))
    return u, 0.0


def ic3d(y, z, h_min, h_max, vel_avg):
    coeff = 2.0
    H = 0.5 * (h_max - h_min)
    r2 = y * y + z * z
    u = vel_avg * coeff * (1.0 - r2 / (H * H))
    return u, 0.0, 0.0


# IC parameters
h_min = -2.0
h_max = 2.2
vel_avg = 3.0

# Compute ic values in 2D
y = 0.7375
u, v = ic2d(y, h_max, h_min, vel_avg)

# Print ic values
print("\n2D laminar flow:")
print("phi: ", 0.0)
print("u: ", u)
print("v: ", v)

# Compute ic values in 3D
z = 0.9775
u, v, w = ic3d(y, z, h_min, h_max, vel_avg)

# Print ic values
print("\n3D laminar flow:")
print("phi: ", 0.0)
print("u: ", u)
print("v: ", v)
print("w: ", w)
