import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from motion_estimation.quaternion import (
    quaternion_to_rotation, rotation_to_quaternion
)


def test_rotation_to_quaternion():
    # TODO add proper testcases
    Q = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1]
    ])
    q = np.array([0, 1, 1, 0]) / np.sqrt(2)
    assert_array_almost_equal(rotation_to_quaternion(Q), q)

    Q = np.array([
        [-1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])
    q = np.array([0, 0, 1, 1]) / np.sqrt(2)
    assert_array_almost_equal(rotation_to_quaternion(Q), q)

    Q = np.array([
        [0, 0, 1],
        [0, -1, 0],
        [1, 0, 0]
    ])
    q = np.array([0, 1, 0, 1]) / np.sqrt(2)
    assert_array_almost_equal(rotation_to_quaternion(Q), q)

    Q = np.eye(3)
    q = np.array([1, 0, 0, 0])
    assert_array_almost_equal(rotation_to_quaternion(Q), q)

    Q = np.array([
        [-0.6, 0, 0.8],
        [0, -1, 0],
        [0.8, 0, 0.6]
    ])
    q = np.array([0, 1, 0, 2]) / np.sqrt(5)
    assert_array_almost_equal(rotation_to_quaternion(Q), q)

    Q = np.array([
        [0.6, -0.8, 0],
        [0.8, 0.6, 0],
        [0, 0, 1]
    ])
    q = np.array([2, 0, 0, 1]) / np.sqrt(5)
    assert_array_almost_equal(rotation_to_quaternion(Q), q)


def test_quaternion_to_rotation():
    q = np.array([1, 0, 0, 0])
    Q = np.eye(3)
    assert_array_almost_equal(quaternion_to_rotation(q), Q)

    Q = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1]
    ])

    q = np.array([0, 1, 1, 0]) / np.sqrt(2)
    assert_array_equal(quaternion_to_rotation(q), Q)

    q = np.array([0, 1, 1, 0])
    assert_array_equal(quaternion_to_rotation(q), Q)

    # random vector
    q = np.array([0.51249408, -0.6836772, -0.99831886, 0.25125051])
    R = quaternion_to_rotation(q)

    assert_array_almost_equal(np.dot(R, R.T), np.eye(3))
    assert_array_almost_equal(np.linalg.det(R), 1)


test_rotation_to_quaternion()
test_quaternion_to_rotation()
