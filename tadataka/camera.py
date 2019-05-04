import numpy as np

from tadataka.utils import to_tuple_if_scalar


class CameraParameters(object):
    def __init__(self, focal_length, offset):
        ox, oy = to_tuple_if_scalar(offset)
        fx, fy = to_tuple_if_scalar(focal_length)

        self.focal_length = np.array([fx, fy])
        self.offset = np.array([ox, oy])

    # TODO add tests
    @property
    def matrix(self):
        K = np.zeros((3, 3))
        K[[0, 1], [0, 1]] = self.focal_length
        K[0:2, 2] = self.offset
        K[2, 2] = 1.0
        return K
