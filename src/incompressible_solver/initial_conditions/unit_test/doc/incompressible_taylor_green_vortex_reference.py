import math

# Node coordinates (set in the unit test)
x_list = [-0.15, -0.05, 0.05, 0.15]
y_list = [-0.05, 0.15, 0.35, 0.55]


# Function
def ic(x, y):
    u = math.cos(x) * math.sin(y)
    v = -math.sin(x) * math.cos(y)
    p = -0.25 * (math.cos(2.0 * x) + math.cos(2.0 * y))
    return u, v, p


# Compute ic values
u_list = []
v_list = []
p_list = []
for x, y in zip(x_list, y_list):
    u, v, p = ic(x, y)
    p_list.append(p)
    u_list.append(u)
    v_list.append(v)

# Print ic values
print("phi: ", p_list)
print("u: ", u_list)
print("v: ", v_list)
