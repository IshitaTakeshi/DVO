import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from numpy.testing import assert_array_equal

from motion_estimation.rigid import transform, transformation_matrix


def test_transform():
    g = np.array([
        [0, 0, -1, 0],
        [-1, 0, 0, -1],
        [0, -1, 0, 2],
        [0, 0, 0, 1]
    ])
    P = np.array([1, 3, 2])
    assert_array_equal(transform(g, P), np.array([-2, -2, -1]))

    g = np.eye(4)
    assert_array_equal(transform(g, P), P)


def test_transformation_matrix():
    omega = np.array([1, 1, 0]) * np.pi / np.sqrt(2)
    nu = np.array([-1, 3, 2])

    K = np.array([
        [0, 0, 1],
        [0, 0, -1],
        [-1, 1, 0]
    ]) * np.pi / np.sqrt(2)

    theta = np.pi

    I = np.eye(3)

    A = np.sin(theta) / theta * K
    B = (1 - np.cos(theta)) / np.power(theta, 2) * np.dot(K, K)
    R = I + A + B

    A = (1 - np.cos(theta)) / np.power(theta, 2) * K
    B = (theta - np.sin(theta)) / np.power(theta, 3) * np.dot(K, K)
    V = I + A + B

    v = np.concatenate((omega, nu))
    g = transformation_matrix(v)

    assert_array_equal(g[0:3, 0:3], R)
    assert_array_equal(g[0:3, 3], np.dot(V, nu))
    assert_array_equal(g[3], np.array([0, 0, 0, 1]))

    omega = np.zeros(3)
    nu = np.array([-1, 3, 2])
    v = np.concatenate((omega, nu))
    g = transformation_matrix(v)
    GT = np.array([
        [1, 0, 0, -1],
        [0, 1, 0, 3],
        [0, 0, 1, 2],
        [0, 0, 0, 1]
    ])

    assert_array_equal(g, GT)


test_transform()
test_transformation_matrix()
