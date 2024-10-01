import numpy as np

# 2D and 3D velocity gradients
grad_vel_2D = np.array([[-0.25, -0.5], [0.5, 1.0]])
grad_vel_3D = np.array([[-0.25, -0.5, -0.125], [0.5, 1.0, 0.25],
                        [-0.75, -1.5, -0.375]])

normals = [-0.02, 0.04, -0.06]

dim_list = [2, 3]

for dim in dim_list:
    print("Computing boundary velocity gradient in ", dim, "D\n")

    grad_vel = grad_vel_2D
    if (dim == 3):
        grad_vel = grad_vel_3D

    normals_dim = normals[:dim]

    for d in range(dim):
        boundary_grad_u = grad_vel[d] - np.array(
            np.dot(grad_vel[d], normals_dim)) * normals_dim
        print("Boundary gradient of component ", d, " is: ", boundary_grad_u,
              "\n")

    print("\n")
