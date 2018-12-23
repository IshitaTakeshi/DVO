import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from numpy.testing import assert_array_equal
import numpy as np

from motion_estimation.twist import cross_product_matrix, twist


def test_cross_product_matrix():
    GT = np.array([
        [0, -3, 2],
        [3, 0, -1],
        [-2, 1, 0]
    ])

    assert_array_equal(cross_product_matrix([1, 2, 3]), GT)


def test_twist():
    GT = np.array([
        [0, -3, 2, 4],
        [3, 0, -1, 5],
        [-2, 1, 0, 6],
        [0, 0, 0, 0]
    ])

    assert_array_equal(twist([1, 2, 3, 4, 5, 6]), GT)


test_cross_product_matrix()
test_twist()
