import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_equal)

from motion_estimation.rigid import transform, transformation_matrix, rodrigues


def test_transform():
    P = np.array([1, 3, 2])

    g = np.array([
        [0, 0, -1, 0],
        [-1, 0, 0, -1],
        [0, -1, 0, 2],
        [0, 0, 0, 1]
    ])
    GT = np.array([-2, -2, -1])
    assert_array_equal(transform(g, P), GT)

    g = np.eye(4)
    assert_array_equal(transform(g, P), P)

    g = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    GT = np.array([1, 4, 2])
    assert_array_equal(transform(g, P), GT)


def test_rodrigues():
    GT = np.array([
        [1,  0,  0],
        [0,  0, -1],
        [0,  1,  0]
    ])
    assert_array_almost_equal(rodrigues([np.pi / 2, 0, 0]), GT)

    GT = np.array([
        [1 / np.sqrt(2), 0, 1 / np.sqrt(2)],
        [0, 1, 0],
        [-1 / np.sqrt(2), 0, 1 / np.sqrt(2)]
    ])
    assert_array_almost_equal(rodrigues([0, np.pi / 4, 0]), GT)

    GT = np.array([
        [1 / np.sqrt(2), 1 / np.sqrt(2), 0],
        [-1 / np.sqrt(2), 1 / np.sqrt(2), 0],
        [0, 0, 1]
    ])
    assert_array_almost_equal(rodrigues([0, 0, -np.pi / 4]), GT)


def test_transformation_matrix():
    # Case 1: Smoke test

    v = np.array([-1, 3, 2])
    omega = np.array([np.pi / 2, 0, 0])
    xi = np.concatenate([v, omega])

    G = transformation_matrix(xi)

    R = np.array([
        [1,  0,  0],
        [0,  0, -1],
        [0,  1,  0]
    ])

    assert_array_almost_equal(G[0:3, 0:3], R)

    # in this case theta == np.pi / 2

    K = np.array([
        [0, 0, 0],
        [0, 0, -np.pi / 2],
        [0, np.pi / 2, 0]
    ])

    I = np.eye(3)
    B = 4 / np.power(np.pi, 2) * K
    C = (4 / np.power(np.pi, 2) - 8 / np.power(np.pi, 3)) * np.dot(K, K)
    V = I + B + C
    assert_array_almost_equal(G[0:3, 3], np.dot(V, v))

    assert_array_almost_equal(G[3], np.array([0, 0, 0, 1]))

    # Case 2: When theta is very small
    # Check that the translation part calculated by the approximated form
    # is close to the result from the result of the usual calculation

    epsilon = np.pi * 1e-7  # very small value

    v = np.array([-1, 3, 2])
    omega = np.array([epsilon, 0, 0])

    xi = np.concatenate([v, omega])
    G = transformation_matrix(xi)

    K = np.array([
        [0, 0, 0],
        [0, 0, -epsilon],
        [0, epsilon, 0]
    ])
    theta = epsilon
    V = (I + (1 - np.cos(theta)) / np.power(theta, 2) * K +
         (theta - np.sin(theta)) / np.power(theta, 3) * np.dot(K, K))
    assert_array_almost_equal(G[0:3, 3], np.dot(V, v), decimal=7)

    # Case 3: theta == 0
    xi = np.array([-1, 3, 2, 0, 0, 0])
    G = transformation_matrix(xi)
    GT = np.array([
        [1, 0, 0, -1],
        [0, 1, 0, 3],
        [0, 0, 1, 2],
        [0, 0, 0, 1]
    ])

    assert_array_equal(G, GT)


test_rodrigues()
test_transform()
test_transformation_matrix()
