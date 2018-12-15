import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal


from motion_estimation.camera import CameraParameters
from motion_estimation.rigid import transformation_matrix, transform
from motion_estimation.twist import twist
from motion_estimation.jacobian import (
        jacobian_transform, jacobian_rigid_motion, jacobian_projections,
        calc_image_gradient)


def stack(K):
    return K[:3].T.flatten()


def test_jacobian_rigid_motion():
    g = np.array([
         [1, 4, 7, 10],
         [2, 5, 8, 11],
         [3, 6, 9, 12],
         [0, 0, 0, 1]
    ])

    GT = np.array([
       [0, 3, -2, 0, 0, 0],
       [-3, 0, 1, 0, 0, 0],
       [2, -1, 0, 0, 0, 0],
       [0, 6, -5, 0, 0, 0],
       [-6, 0, 4, 0, 0, 0],
       [5, -4, 0, 0, 0, 0],
       [0, 9, -8, 0, 0, 0],
       [-9, 0, 7, 0, 0, 0],
       [8, -7, 0, 0, 0, 0],
       [0, 12, -11, 1, 0, 0],
       [-12, 0, 10, 0, 1, 0],
       [11, -10, 0, 0, 0, 1]
    ])

    MG = jacobian_rigid_motion(g)

    assert_array_equal(MG, GT)

    k = [1, 2, 3, 4, 5, 6]
    K = twist(k)[:3]
    assert_array_equal(stack(np.dot(K, g)), np.dot(MG, k))


def test_jacobian_transform():
    GT = np.array([
        [[1, 0, 0, 2, 0, 0, 3, 0, 0, 1, 0, 0],
         [0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 1, 0],
         [0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 1]],
        [[5, 0, 0, 4, 0, 0, 8, 0, 0, 1, 0, 0],
         [0, 5, 0, 0, 4, 0, 0, 8, 0, 0, 1, 0],
         [0, 0, 5, 0, 0, 4, 0, 0, 8, 0, 0, 1]]
    ])
    P = np.array([
        [1, 2, 3],
        [5, 4, 8]
    ])

    assert_array_equal(jacobian_transform(P), GT)

    g = transformation_matrix(
        np.array([0.1, -0.5, 0.4, 0.2, 0.8, -0.3])
    )

    JP = jacobian_transform(P)

    assert_array_almost_equal(np.dot(JP[0], stack(g[:3])), transform(g, P[0]))
    assert_array_almost_equal(np.dot(JP[1], stack(g[:3])), transform(g, P[1]))


def test_jacobian_projections():
    GS = np.arange(24).reshape(8, 3)

    fx = 1.2
    fy = 1.0

    camera_parameters = CameraParameters(
        focal_length=[fx, fy],
        offset=[0, 0]
    )

    JS = jacobian_projections(camera_parameters, GS)

    for J, G in zip(JS, GS):
        x, y, z = G
        GT = np.array([
            [fx / z, 0, -fx * x / pow(z, 2)],
            [0, fy / z, -fy * y / pow(z, 2)]
        ])
        assert_array_almost_equal(J, GT)


def test_calc_image_gradient():
    image = np.array([
        [3, 1, -1, 4],
        [2, 4, 1, -8],
        [-1, 0, 1, 4],
        [7, -3, 0, 5]
    ])

    D = calc_image_gradient(image)

    GT = np.array([
        [2, 2, -5, 1],
        [-2, 3, 9, -10],
        [-1, -1, -3, 5],
        [10, -3, -5, -2]
    ])
    GT = GT.flatten()
    assert_array_equal(D[:, 0], GT)

    GT = np.array([
        [1, -3, -2, 12],
        [3, 4, 0, -12],
        [-8, 3, 1, -1],
        [4, -4, 1, 1]
    ])
    GT = GT.flatten()
    assert_array_equal(D[:, 1], GT)


def test_approximation():
    xi = np.array([1.8, -0.6, 0.9, 1.8, -2.8, -0.6])
    dxi = np.array([4, -2, 2, -2, -3, -2]) / 100

    xi0 = xi
    xi1 = xi + dxi

    S = np.array([
        [1.0, 2.0, -1.0],
        [0.5, -0.4, 0.2]
    ])

    g0 = transformation_matrix(xi0)
    g1 = transformation_matrix(xi1)

    G0 = transform(g0, S)
    G1 = transform(g1, S)

    # U.shape = (n_image_pixels, 3, 12)
    U = jacobian_transform(G0)  # dG / d(stack(g))

    print("dot(U, g1 - g0): ", U.dot(stack(g1) - stack(g0)))
    print("G1 - G0        : ", G1 - G0)


test_jacobian_rigid_motion()
test_jacobian_transform()
test_jacobian_projections()
test_calc_image_gradient()
test_approximation()
