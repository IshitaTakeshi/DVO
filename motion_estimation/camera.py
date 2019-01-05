import numpy as np

from motion_estimation.utils import to_tuple_if_scalar


class CameraParameters(object):
    def __init__(self, focal_length, offset):
        ox, oy = to_tuple_if_scalar(offset)
        fx, fy = to_tuple_if_scalar(focal_length)

        self.focal_length = np.array([fx, fy])
        self.offset = np.array([ox, oy])
