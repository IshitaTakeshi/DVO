import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from numpy.testing import assert_array_equal
import numpy as np

from motion_estimation.visual_odometry import (CameraParameters,
        inverse_projection, transform, jacobian_3dpoint, jacobian_rigid_motion)


def test_inverse_projection():
    camera_parameters = CameraParameters(10, [2, 3])
    S = inverse_projection(
        camera_parameters,
        np.array([5, 2]),
        3
    )

    assert_array_equal(S, np.array([2.1, 1.5, 3]))


def test_transform():
    g = np.array([
        [0, 0, -1, 0],
        [-1, 0, 0, -1],
        [0, -1, 0, 2],
        [0, 0, 0, 1]
    ])
    P = np.array([1, 3, 2])
    assert_array_equal(transform(g, P), np.array([-2, -2, -1]))


def test_jacobian_3dpoint():
    x, y, z = 1, 2, 3
    GT = [
        [x, 0, 0, y, 0, 0, z, 0, 0, 1, 0, 0],
        [0, x, 0, 0, y, 0, 0, z, 0, 0, 1, 0],
        [0, 0, x, 0, 0, y, 0, 0, z, 0, 0, 1]
    ]
    assert_array_equal(jacobian_3dpoint([x, y, z]), GT)


def test_jacobian_rigid_motion():
    g = np.array([
         [1, 2, 3, 10],
         [4, 5, 6, 11],
         [7, 8, 9, 12]
    ])

    GT = np.array([
       [0, 7, -4, 0, 0, 0],
       [-7, 0, 1, 0, 0, 0],
       [4, -1, 0, 0, 0, 0],
       [0, 8, -5, 0, 0, 0],
       [-8, 0, 2, 0, 0, 0],
       [5, -2, 0, 0, 0, 0],
       [0, 9, -6, 0, 0, 0],
       [-9, 0, 3, 0, 0, 0],
       [6, -3, 0, 0, 0, 0],
       [0, 12, -11, 1, 0, 0],
       [-12, 0, 10, 0, 1, 0],
       [11, -10, 0, 0, 0, 1]
    ])

    assert_array_equal(jacobian_rigid_motion(g), GT)

test_inverse_projection()
test_transform()
test_jacobian_3dpoint()
test_jacobian_rigid_motion()
