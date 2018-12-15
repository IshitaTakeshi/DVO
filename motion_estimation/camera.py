import numpy as np

from motion_estimation.utils import to_tuple_if_scalar


class CameraParameters(object):
    def __init__(self, focal_length, offset):
        ox, oy = to_tuple_if_scalar(offset)
        fx, fy = to_tuple_if_scalar(focal_length)

        self.matrix = np.array([
            [fx, 0, -ox],
            [0, fy, -oy],
            [0, 0, 1]
        ])

        self.focal_length = np.array([fx, fy])
        self.offset = np.array([ox, oy])
