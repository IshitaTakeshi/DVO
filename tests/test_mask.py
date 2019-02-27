import numpy as np
from numpy.testing import assert_array_equal

from motion_estimation.mask import is_in_rage, compute_mask


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

    # x and y have to be in range
    # 0 <= x <= 12.0, 0 <= y <= 14.0 respectively
    # otherwise map_coordinates returns nan

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
    depth_map = np.array([
        # True False False True
        [3.1, 0.0, -0.1, 0.4],
        # True True False True
        [2.2, 2.0, -0.8, 0.8]
    ])

    # depth_map.shape == (2, 4)
    # therefore x and y have to be in range
    # 0 <= x <= 3.0, 0 <= y <= 1.0 respectively
    pixel_coordinates = np.array([
        # False True True False
        [0.0, -0.2], [0.0, 0.0], [0.8, 0.2], [1.2, 1.4],
        # False True True False
        [3.1, 2.0], [3.0, 1.0], [1.0, 0.0], [1.9, 2.9]
    ])

    GT = np.array([
        [False, False, False, False],
        [False, True, False, False]
    ])

    mask = compute_mask(depth_map, pixel_coordinates)

    assert_array_equal(mask, GT)


test_is_in_rage()
test_compute_mask()
