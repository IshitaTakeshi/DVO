import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from numpy.testing import assert_array_equal

from motion_estimation.mask import is_in_rage


def test_is_in_rage():
    P = np.array([
        #   x     y
        [-0.8, -0.2],
        [-0.8, 0.2],
        [0.8, -0.2],
        [0.0, -0.2],
        [-0.8, 0.0],
        [0.0, 0.0],
        [0.8, 0.2],
        [11.1, 13.2],
        [12.0, 13.2],
        [11.1, 14.0],
        [12.0, 14.0],
        [12.0, 14.2],
        [12.1, 14.0],
        [12.1, 14.2],
        [13.1, 15.2]
    ])

    image_shape = (15, 13)  # (height, width)

    # cannot be y > 14.0, x > 12.0
    # because in that case map_coordinates returns nan

    GT = np.array([
        [0.0, 0.0],
        [0.8, 0.2],
        [11.1, 13.2],
        [12.0, 13.2],
        [11.1, 14.0],
        [12.0, 14.0],
    ])

    mask = is_in_rage(image_shape, P)

    assert_array_equal(P[mask], GT)


def test_compute_mask():
    pass


test_is_in_rage()
test_compute_mask()
