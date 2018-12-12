import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal


from motion_estimation.camera import CameraParameters
from motion_estimation.rigid import transformation_matrix
from motion_estimation.jacobian import (
        jacobian_transform, jacobian_rigid_motion, jacobian_projections,
        calc_image_gradient)


def test_jacobian_rigid_motion():
    g = np.array([
         [1, 4, 7, 10],
         [2, 5, 8, 11],
         [3, 6, 9, 12]
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

    assert_array_equal(jacobian_rigid_motion(g), GT)


def test_jacobian_transform():
    def homogeneous(p):
        return np.hstack((p, 1))

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

    xi = np.array([0.1, -0.5, 0.4, 0.2, 0.8, -0.3])
    g = transformation_matrix(xi)

    JP = jacobian_transform(P)

    assert_array_equal(
        np.dot(JP[0], g[:3].T.flatten()),
        np.dot(g[:3], homogeneous(P[0]))
    )

    assert_array_equal(
        np.dot(JP[1], g[:3].T.flatten()),
        np.dot(g[:3], homogeneous(P[1]))
    )


def test_jacobian_projections():
    GS = np.arange(24).reshape(8, 3)

    fx = 1.2
    fy = 1.0
    s = 0.8

    camera_parameters = CameraParameters(
        focal_length=[fx, fy],
        offset=[0, 0],
        skew=s
    )

    JS = jacobian_projections(camera_parameters, GS)
    for J, G in zip(JS, GS):
        GT = np.array([
            [fx / G[2], s / G[2], -(fx * G[0] + s * G[1]) / pow(G[2], 2)],
            [0, fy / G[2], -fy * G[1] / pow(G[2], 2)]
        ])
        assert_array_almost_equal(J, GT)


def test_jacobian_rigid_motion():
    g = np.array([
         [1, 4, 7, 10],
         [2, 5, 8, 11],
         [3, 6, 9, 12]
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

    assert_array_equal(jacobian_rigid_motion(g), GT)


test_jacobian_3dpoints()
test_jacobian_rigid_motion()
test_jacobian_projections()
