# Coefficients
nu = 50
rho = 1.0
k = 0.5
cp = 100
h_min = -4.0
h_max = 4.0
T_l = 305.0
T_u = 300.0
S_u = 24000

# Coordinates
x = 2.25
y = 2.20

# Exact solution calculations
H = (h_max - h_min) / 2.0
dT = T_l - T_u
U_avg = S_u * H * H / 3 / nu
Pr = cp * rho * nu / k
E = U_avg * U_avg / cp / dT

## Velocity
u = 3.0 / 2.0 * U_avg * (1.0 - pow(y / H, 2.0))
v = 0.0

print("x velocity: ", u)
print("y velocity: ", v)

## Temperature
T_star = 0.5 * (1.0 - (y / H)) + 3.0 / 4.0 * Pr * E * (1.0 - pow(y / H, 4.0))

temp = T_star * dT + T_u

print("Temperature: ", temp)
