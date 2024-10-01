from math import *

# gas properties
gamma = 1.5
univ_gas_const = 2.0
mmw = 3.0
R = univ_gas_const / mmw
one_over_gm1 = 1.0 / (gamma - 1.0)

# coefficient of sinusoidal mms function
rho_coeff = [0.0125, 0.25, 0.125, 0.5, 0.0, 1.0]
u_coeff = [0.0125, 0.125, 0.125, 0.0, 0.25, 0.08]
v_coeff = [0.0375, 0.25, 0.375, 0.0, 0.5, 1.125]
w_coeff = [0.025, 0.125, 0.25, 1.0, 0.25, 0.0]
T_coeff = [0.0625, 0.375, 0.25, 0.25, 0.5, 1.0]

# evaluation points
x = 0.5
y = 0.5

# mms solution
# f = A * sin( 2*pi*f_x * ( x - phi_x ) ) * sin( 2*pi*f_y * ( y - phi_y ) ) + B
rho = rho_coeff[0] * sin(2.0 * pi * rho_coeff[1] * (x - rho_coeff[3])) * sin(
    2.0 * pi * rho_coeff[2] * (y - rho_coeff[4])) + rho_coeff[5]

u = u_coeff[0] * sin(2.0 * pi * u_coeff[1] *
                     (x - u_coeff[3])) * sin(2.0 * pi * u_coeff[2] *
                                             (y - u_coeff[4])) + u_coeff[5]

v = v_coeff[0] * sin(2.0 * pi * v_coeff[1] *
                     (x - v_coeff[3])) * sin(2.0 * pi * v_coeff[2] *
                                             (y - v_coeff[4])) + v_coeff[5]

T = T_coeff[0] * sin(2.0 * pi * T_coeff[1] *
                     (x - T_coeff[3])) * sin(2.0 * pi * T_coeff[2] *
                                             (y - T_coeff[4])) + T_coeff[5]

print("rho: ", rho)
print("u: ", u)
print("v: ", v)
print("T: ", T)
