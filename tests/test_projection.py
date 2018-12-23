import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from numpy.testing import assert_array_equal, assert_array_almost_equal
import numpy as np

from motion_estimation.camera import CameraParameters
from motion_estimation.projection import inverse_projection, projection


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
    depth = np.array([1, 2, 2, 1, 2, 3])

    S = inverse_projection(camera_parameters, P, depth)

    GT = np.array([
        [(0+1)*1 / 2, (0-2)*1 / 2, 1],
        [(0+1)*2 / 2, (1-2)*2 / 2, 2],
        [(0+1)*2 / 2, (2-2)*2 / 2, 2],
        [(1+1)*1 / 2, (0-2)*1 / 2, 1],
        [(1+1)*2 / 2, (1-2)*2 / 2, 2],
        [(1+1)*3 / 2, (2-2)*3 / 2, 3]
    ])

    assert_array_equal(S, GT)

    # is really the inverse
    assert_array_equal(P, projection(camera_parameters, S))


def test_projection():
    camera_parameters = CameraParameters([12, 16], [3, 4])

    S = np.array([
        [1, 2, 3],
        [4, 5, 2]
    ])

    P = projection(camera_parameters, S)

    GT = np.array([
        [1 *12 / 3 - 3, 2 * 16 / 3 - 4],
        [4 *12 / 2 - 3, 5 * 16 / 2 - 4]
    ])

    assert_array_almost_equal(P, GT)

    depth = S[0:2, 2]
    Q = inverse_projection(camera_parameters, P, depth)

    assert_array_almost_equal(Q, S)


test_inverse_projection()
test_projection()
