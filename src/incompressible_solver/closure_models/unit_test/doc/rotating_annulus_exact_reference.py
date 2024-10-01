import numpy as np

# Coefficients
nu = 50
rho = 1.0
k = 0.5
Ro = 4.0
Ri = 2.0
omega = 0.5
To = 274.0
Ti = 273.0

# Coordinates
x = 2.25
y = 2.20

# Exact solution calculations
r = np.sqrt(x * x + y * y)
xi = r / Ro
kappa = Ri / Ro

## Velocity
u_phi = omega * Ro * Ro / (Ro * Ro - Ri * Ri) * (r - (Ri * Ri) / r)
u = -u_phi * y / r
v = u_phi * x / r

print("x velocity: ", u)
print("y velocity: ", v)

## Temperature
N = nu * rho * omega * omega * Ro * Ro / k / (To - Ti) * pow(kappa, 4.0) / pow(
    1.0 - pow(kappa, 2.0), 2.0)

theta = (1.0 - np.log(xi) / np.log(kappa)) + N * (
    (1 - 1 / (xi * xi)) - (1 - 1 /
                           (kappa * kappa)) * np.log(xi) / np.log(kappa))

temp = theta * (To - Ti) + Ti

print("Temperature: ", temp)
