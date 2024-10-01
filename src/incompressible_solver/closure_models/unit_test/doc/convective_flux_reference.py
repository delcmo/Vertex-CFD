import numpy as np

rho = 1.0
Cp = 5.0
u = 0.25
v = 0.5
w = 0.125
p = 0.75  # lagrange pressure
T = u + v

print("\nFlux with scaled density:\n")
fc_continuity = rho * np.array([u, v, w])
print(fc_continuity)

fc_momentum_0 = rho * np.array([u * u + p, u * v, u * w])
print(fc_momentum_0)

fc_momentum_1 = rho * np.array([u * v, v * v + p, v * w])
print(fc_momentum_1)

fc_momentum_2 = rho * np.array([u * w, v * w, w * w + p])
print(fc_momentum_2)

fc_energy = rho * Cp * np.array([u * T, v * T, w * T])
print(fc_energy)

print("\nFlux with unscaled density:\n")
rho = 3.0
fc_continuity = rho * np.array([u, v, w])
print(fc_continuity)

fc_momentum_0 = np.array([rho * u * u + p, rho * u * v, rho * u * w])
print(fc_momentum_0)

fc_momentum_1 = np.array([rho * u * v, rho * v * v + p, rho * v * w])
print(fc_momentum_1)

fc_momentum_2 = np.array([rho * u * w, rho * v * w, rho * w * w + p])
print(fc_momentum_2)

fc_energy = rho * Cp * np.array([u * T, v * T, w * T])
print(fc_energy)
