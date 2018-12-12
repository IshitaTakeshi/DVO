import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from numpy.testing import assert_array_equal

from motion_estimation.rigid import transform


def test_transform():
    g = np.array([
        [0, 0, -1, 0],
        [-1, 0, 0, -1],
        [0, -1, 0, 2],
        [0, 0, 0, 1]
    ])
    P = np.array([1, 3, 2])
    assert_array_equal(transform(g, P), np.array([-2, -2, -1]))


test_transform()
