# Functions
def bc2d(y, h, U_0):
    u = U_0 * pow(1.0 - pow(y / h, 18.0), 2.0)
    return u, 0.0


def bc3d(y, z, h, U_0):
    u = U_0 * pow(1.0 - pow(y / h, 18.0), 2.0) * pow(1.0 - pow(z / h, 18.0),
                                                     2.0)
    return u, 0.0, 0.0


# IC parameters
h = 1.0
U_0 = 2.0

# Compute ic values in 2D
y = 0.7375
u, v = bc2d(y, h, U_0)

# Print ic values
print("\n2D laminar flow:")
print("u: ", u)
print("v: ", v)

# Compute ic values in 3D
z = 0.9775
u, v, w = bc3d(y, z, h, U_0)

# Print ic values
print("\n3D laminar flow:")
print("u: ", u)
print("v: ", v)
print("w: ", w)
