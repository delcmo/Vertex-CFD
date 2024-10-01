import math

# Node coordinates (set in the unit test)
x_list = [-0.15, -0.05, 0.05, 0.15]
y_list = [-0.05, 0.15, 0.35, 0.55]


# Function
def ic(x, y, time, nu):
    Ft = math.exp(-2.0 * nu * time)
    u = math.cos(x) * math.sin(y) * Ft
    v = -math.sin(x) * math.cos(y) * Ft
    p = -0.25 * (math.cos(2.0 * x) + math.cos(2.0 * y)) * Ft * Ft
    return u, v, p


# Compute ic values
time = 0.5
nu = 0.325
p_list = []
u_list = []
v_list = []
for x, y in zip(x_list, y_list):
    u, v, p = ic(x, y, time, nu)
    p_list.append(p)
    u_list.append(u)
    v_list.append(v)

# Print ic values
print("phi: ", p_list)
print("u: ", u_list)
print("v: ", v_list)
