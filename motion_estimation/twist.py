import numpy as np


def cross_product_matrix(k):
    return np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])


def hat(k):
    K = np.empty((4, 4))
    K[0:3, 0:3] = cross_product_matrix(k[:3])
    K[0:3, 3] = k[3:]
    K[3] = 0
    return K
