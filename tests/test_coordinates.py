from numpy.testing import assert_array_equal
import numpy as np

from tadataka.coordinates import compute_pixel_coordinates


def test_compute_pixel_coordinates():
    GT = np.array([
    #    x  y
        [0, 0],
        [1, 0],
        [2, 0],
        [0, 1],
        [1, 1],
        [2, 1],
        [0, 2],
        [1, 2],
        [2, 2],
        [0, 3],
        [1, 3],
        [2, 3]
    ])

    height = 4
    width = 3
    assert_array_equal(compute_pixel_coordinates((height, width)), GT)

    compute_pixel_coordinates((height, width))

test_compute_pixel_coordinates()
