import numpy as np
from motion_estimation.rigid import exp_so3, tangent_so3


xi = np.array([1.8, -0.6, 0.9])
dxi = np.array([2e-2, -2e-2, 2e-2])

xi0 = xi
xi1 = xi + dxi
R0 = exp_so3(xi0)
R1 = exp_so3(xi1)

np.set_printoptions(suppress=True, precision=4)
print(tangent_so3(xi0))
print(R0.dot(R0.T))

print("tangent_so3(xi0).dot(R0)")
print(tangent_so3(xi0).dot(R0))
print("R1 - R0")
print(R1 - R0)
print((R1 - R0) / tangent_so3(xi0).dot(R0))
