nu = 0.25
cn1 = 16.0
sigma = 2.0 / 3.0

# Negative SA variable
sa_var = -3.0
xi = sa_var / nu
xi3 = xi * xi * xi
f_n = (cn1 + xi3) / (cn1 - xi3)

print("Negative SA variable:", (nu + sa_var * f_n) / sigma)

# Positive SA variable
sa_var = 3.0
xi = sa_var / nu
xi3 = xi * xi * xi
f_n = 1.0

print("Positive SA variable:", (nu + sa_var * f_n) / sigma)
