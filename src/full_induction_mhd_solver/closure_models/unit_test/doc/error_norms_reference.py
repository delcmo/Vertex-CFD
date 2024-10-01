import math

# computational/exact solution
comp = [0.3, 0.4, 0.5]
exact = [0.7, 0.9, 1.1]

var_id = range(len(comp))

# L1 error norm
print("L1 norm of vec{B}):")
for i in var_id:
    print(abs(comp[i] - exact[i]))

# L2 error norm
print("L2 norm vec{B}:")
for i in var_id:
    print((comp[i] - exact[i])**2.0)
