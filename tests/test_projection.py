import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from numpy.testing import assert_array_equal
import numpy as np

from motion_estimation.camera import CameraParameters
from motion_estimation.projection import inverse_projection


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


test_inverse_projection()
