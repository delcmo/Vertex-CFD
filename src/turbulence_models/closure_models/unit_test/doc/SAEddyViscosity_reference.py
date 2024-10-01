cv1 = 7.1
nu = 0.25

# Negative SA variable with or without temperature equation
sa_var = -3.0

print("Negative SA variable:", 0.0)

# Positive SA variable but larger than max tolerance without temperature equation
sa_var = 3.0
xi = sa_var / nu
xi3 = xi * xi * xi
f_v1 = xi3 / (xi3 + cv1 * cv1 * cv1)

print("Positive SA variable (larger then tolerrance). SA Eddy viscosity is:",
      sa_var * f_v1)
