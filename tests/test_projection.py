import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from numpy.testing import assert_array_equal, assert_array_almost_equal
import numpy as np

from motion_estimation.visual_odometry import (CameraParameters,
        inverse_projection, transform, jacobian_3dpoints,
        jacobian_rigid_motion, jacobian_projections)


def test_inverse_projection():
    camera_parameters = CameraParameters(focal_length=[2, 2], offset=[1, -2])
    P = np.array([
        [0, 0],
        [0, 1],
        [0, 2],
        [1, 0],
        [1, 1],
        [1, 2],
    ])
    depth = np.array([1, 2, 0, 1, 2, 3])

    S = inverse_projection(camera_parameters, P, depth)

    GT = np.array([
        [(0+1)*1 / 2, (0-2)*1 / 2, 1],
        [(0+1)*2 / 2, (1-2)*2 / 2, 2],
        [(0+1)*0 / 2, (2-2)*0 / 2, 0],
        [(1+1)*1 / 2, (0-2)*1 / 2, 1],
        [(1+1)*2 / 2, (1-2)*2 / 2, 2],
        [(1+1)*3 / 2, (2-2)*3 / 2, 3]
    ])

    assert_array_equal(S, GT)


def test_transform():
    g = np.array([
        [0, 0, -1, 0],
        [-1, 0, 0, -1],
        [0, -1, 0, 2],
        [0, 0, 0, 1]
    ])
    P = np.array([1, 3, 2])
    assert_array_equal(transform(g, P), np.array([-2, -2, -1]))


def test_jacobian_3dpoints():
    x, y, z = 1, 2, 3
    GT = np.array([
        [[x, 0, 0, y, 0, 0, z, 0, 0, 1, 0, 0],
         [0, x, 0, 0, y, 0, 0, z, 0, 0, 1, 0],
         [0, 0, x, 0, 0, y, 0, 0, z, 0, 0, 1]],
        [[x, 0, 0, y, 0, 0, z, 0, 0, 1, 0, 0],
         [0, x, 0, 0, y, 0, 0, z, 0, 0, 1, 0],
         [0, 0, x, 0, 0, y, 0, 0, z, 0, 0, 1]]
    ])
    P = np.array([
        [x, y, z],
        [x, y, z]
    ])
    assert_array_equal(jacobian_3dpoints(P), GT)


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


test_inverse_projection()
test_transform()
test_jacobian_3dpoints()
test_jacobian_rigid_motion()
test_jacobian_projections()
