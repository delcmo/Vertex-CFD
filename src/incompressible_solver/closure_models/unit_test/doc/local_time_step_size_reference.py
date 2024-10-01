# Dependent values
u = -1.5
v = 2.0
w = -0.5
h0 = 0.25
h1 = 0.5
h2 = 0.75

# 2-D case
dt = 1.0 / (abs(u) / h0 + abs(v) / h1)
print("2-D dt: ", dt)

# 3-D case
dt = 1.0 / (abs(u) / h0 + abs(v) / h1 + abs(w) / h2)
print("3-D dt: ", dt)
