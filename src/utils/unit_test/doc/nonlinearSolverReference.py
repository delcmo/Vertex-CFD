from scipy.optimize import fsolve
import math


##---------------------------------------------------------------------------##
def quadratic2d(p):
    x, y = p
    return (y * y - x + 1, x * x - y - 1)


x, y = fsolve(quadratic2d, (1, 1), xtol=1.0e-13)

print(x, y)
print(quadratic2d((x, y)))


##---------------------------------------------------------------------------##
def linear3d(p):
    x, y, z = p
    return (x - y + z - 1, 2 * x - z, x + 0.5 * y + 0.5 * z - 3)


x, y, z = fsolve(linear3d, (1, 1, 1), xtol=1.0e-13)

print(x, y, z)
print(linear3d((x, y, z)))


##---------------------------------------------------------------------------##
def quadratic3d(p):
    x, y, z = p
    return (x * y * z + y - 1, y * y + x * x - 3, x * z + y - 5)


x, y, z = fsolve(quadratic3d, (1, 1, 1), xtol=1.0e-13)

print(x, y, z)
print(quadratic3d((x, y, z)))

##---------------------------------------------------------------------------##
