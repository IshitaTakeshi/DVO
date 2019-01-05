import numpy as np
from motion_estimation.twist import cross_product_matrix


def rodrigues(omega):
    I = np.eye(3)
    theta = np.linalg.norm(omega)

    if np.isclose(theta, 0):
        return I

    omega = omega / theta
    K = cross_product_matrix(omega)
    cos = np.cos(theta)
    sin = np.sin(theta)
    return I * cos + (1 - cos) * np.outer(omega, omega) + sin * K


def transform(G, P):
    """

    .. math::
        G = \\begin{bmatrix}
            R & T \\\\
            \\mathbf{0}^{\\top} & 1 \\\\
        \\end{bmatrix}

    :math:`RP + T`
    """

    P = np.dot(G[0:3, 0:3], P.T)
    return P.T + G[0:3, 3]


def transformation_matrix(v):
    omega, t = v[:3], v[3:]

    R = rodrigues(omega)

    theta = np.linalg.norm(omega)
    K = cross_product_matrix(omega)
    I = np.eye(3)

    if np.isclose(theta, 0):
        R = V = rodrigues(omega)
    else:
        # TODO not necessary to square
        V = I + ((1 - np.cos(theta)) / pow(theta, 2) * K +
                 (theta - np.sin(theta)) / pow(theta, 3) * np.dot(K, K))
        R = I + (np.sin(theta) / theta * K +
                 (1 - np.cos(theta)) / pow(theta, 2) * np.dot(K, K))

    g = np.empty((4, 4))
    g[0:3, 0:3] = R
    g[0:3, 3] = np.dot(V, t)
    g[3, 0:3] = 0
    g[3, 3] = 1
    return g
