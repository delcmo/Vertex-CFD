import math

pi = math.pi

# Node coordinates (set in the unit test)
x_list = [-0.15, -0.05, 0.05, 0.15]
y_list = [-0.05, 0.15, 0.35, 0.55]


# Function
def ic(x, y):
    u = -2.0 * math.cos(pi * y) * math.sin(pi * y) * math.sin(
        pi * x) * math.sin(pi * x)
    v = 2.0 * math.cos(pi * x) * math.sin(pi * x) * math.sin(
        pi * y) * math.sin(pi * y)
    return u, v


# Compute ic values
u_list = []
v_list = []
for x, y in zip(x_list, y_list):
    u, v = ic(x, y)
    u_list.append(u)
    v_list.append(v)

# Print ic values
print("phi: ", 0.0)
print("u: ", u_list)
print("v: ", v_list)
