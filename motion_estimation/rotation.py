import numpy as np

from motion_estimation.twist import skew_matrix


def rodrigues(v):
    I = np.eye(3)
    theta = np.linalg.norm(v)

    if np.isclose(theta, 0):
        return I

    v = v / theta
    K = skew_matrix(v)
    cos = np.cos(theta)
    sin = np.sin(theta)
    return I * cos + (1 - cos) * np.outer(v, v) + sin * K
