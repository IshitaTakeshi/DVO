import numpy as np
from numpy.linalg import norm
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_equal)

from scipy.linalg import expm

from tadataka.rigid import (exp_so3, exp_se3, log_so3, log_se3,
                            tangent_so3, tangent_se3, transform)


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


def test_tangent_so3():
    GT = np.array([
        [0, -3, 2],
        [3, 0, -1],
        [-2, 1, 0]
    ])
    assert_array_equal(tangent_so3([1, 2, 3]), GT)


def test_tangent_se3():
    GT = np.array([
        [0, -6, 5, 1],
        [6, 0, -4, 2],
        [-5, 4, 0, 3],
        [0, 0, 0, 0]
    ])
    assert_array_equal(tangent_se3([1, 2, 3, 4, 5, 6]), GT)


def test_exp_so3():
    def run(omega):
        assert_array_almost_equal(
            exp_so3(omega),
            expm(tangent_so3(omega))
        )

    run([0, 0, 0])
    run(np.array([2e-9, 0, 1e-9]))
    run(np.array([1, -1, 0]) * np.pi)
    run(np.array([-1 / 2, 1 / 4, -3 / 4])  * np.pi)


def test_exp_se3():
    def run(xi):
        assert_array_almost_equal(
            exp_se3(xi),
            expm(tangent_se3(xi))
        )

    run(np.array([1, 2, -3, 0, 0, 0]))
    run(np.array([1, 2, -3, 0, 0, 1e-9]))
    run(np.array([1, -1, 2, np.pi / 2, 0, 0]))
    run(np.array([-1, 2, 1, 0, -np.pi / 2, np.pi / 4]))


def test_log_so3():
    def run(omega):
        # test log(exp(omega)) == omega
        omega_pred, theta_pred = log_so3(expm(tangent_so3(omega)))
        assert_array_almost_equal(omega_pred * theta_pred, omega)

    run([0, 0, 0])
    run(np.array([1 / 2, 0, -1 / 4]) * np.pi)
    run(np.array([-1 / 2, 1 / 4, -3 / 4])  * np.pi)


def test_log_se3():
    def run(xi):
        # test log(exp(xi)) == xi
        assert_array_almost_equal(log_se3(expm(tangent_se3(xi))), xi)

    run(np.array([1, 2, -3, 0, 0, 0]))
    run(np.array([1, -1, 2, np.pi / 2, 0, 0]))
    run(np.array([-1, 2, 1, 0, -np.pi / 2, np.pi / 4]))
