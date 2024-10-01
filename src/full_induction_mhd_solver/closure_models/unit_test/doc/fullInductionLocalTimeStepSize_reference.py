import math

# Dependent values
u = -1.5
v = 2.0
w = -0.5

b_0 = 1.1
b_1 = -1.2
b_2 = 1.3

rho = 0.9
mu_0 = 0.1

# Alfven speeds
c_a_0 = abs(b_0) / math.sqrt(mu_0 * rho)
c_a_1 = abs(b_1) / math.sqrt(mu_0 * rho)
c_a_2 = abs(b_2) / math.sqrt(mu_0 * rho)

h0 = 0.25
h1 = 0.5
h2 = 0.75

# 2-D case, c_h = 0.1
c_h = 0.1
dt = 1.0 / ((abs(u) + max(c_a_0, c_h)) / h0 + (abs(v) + max(c_a_1, c_h)) / h1)
print("2-D dt, c_h = 0.1: ", dt)

# 3-D case, large c_h
c_h = 5.0
dt = 1.0 / ((abs(u) + max(c_a_0, c_h)) / h0 + (abs(v) + max(c_a_1, c_h)) / h1 +
            (abs(w) + max(c_a_2, c_h)) / h2)
print("3-D dt, c_h = 5.0: ", dt)

# 3-D case, no c_h contribution
c_h = 5.0
dt = 1.0 / ((abs(u) + c_a_0) / h0 + (abs(v) + c_a_1) / h1 +
            (abs(w) + c_a_2) / h2)
print("3-D dt, no c_h: ", dt)
