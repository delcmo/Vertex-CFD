import math

#
x = 0.2
y = 0.4
x0 = 0.1
y0 = -0.2
dx = x - x0
dy = y - y0

#
u0 = 2.0
v0 = -0.5

#
time = 0.6

#
r = (dx - time * u0) * (dx - time * u0)
r += (dy - time * u0) * (dy - time * u0)
r = math.sqrt(r)
r2 = r * r

#
lp = 1.0 + 0.5 * math.exp(1.0) * (1.0 - r2 * math.exp(-r2))
print("Exact lagrange pressure:", lp)

mg0 = -math.exp(0.5 * (1.0 - r2)) * dy
mg1 = math.exp(0.5 * (1.0 - r2)) * dx
print("Exact x-magnetic field:", mg0)
print("Exact y-magnetic field:", mg1)

v0 = mg0 + u0
v1 = mg1
print("Exact x-velocity:", v0)
print("Exact y-velocity:", v1)
