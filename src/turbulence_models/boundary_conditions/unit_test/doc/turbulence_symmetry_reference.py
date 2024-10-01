import numpy as np

grad_var = [0.02, 0.04, 0.06]
normals = [0.33, 0.66, 0.99]

dim_list = [2, 3]

for dim in dim_list:
    print("Computing boundary variable gradient in ", dim, "D\n")

    normals_dim = normals[:dim]
    grad_var_dim = grad_var[:dim]

    boundary_grad_var = grad_var_dim - np.array(
        np.dot(grad_var_dim, normals_dim)) * normals_dim

    for d in range(dim):
        print("Boundary gradient component ", d, " = ", boundary_grad_var[d],
              "\n")

    print("\n")
