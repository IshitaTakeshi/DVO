import numpy as np
from motion_estimation.twist import cross_product_matrix

# Reference:
# Eade Ethan. "Lie groups for 2d and 3d transformations."
# http://ethaneade.com/lie.pdf
# See Section 3.2 Exponential Map


# threshold to use an approximated form
epsilon = 1e-6


def rodrigues(omega):
    I = np.eye(3)
    theta = np.linalg.norm(omega)

    if theta < epsilon:  # since theta = norm(omega) >= 0
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


def transformation_matrix(xi):
    v, omega = xi[:3], xi[3:]

    K = cross_product_matrix(omega)
    I = np.eye(3)
    R = rodrigues(omega)

    theta2 = np.dot(omega, omega)
    theta = np.sqrt(theta2)

    if theta < epsilon:  # since theta = norm(omega) >= 0
        # teylor expansion of V
        V = I + K / 2.0 + np.dot(K, K) / 6.0
    else:
        theta3 = theta2 * theta
        V = (I + (1 - np.cos(theta)) / theta2 * K +
             (theta - np.sin(theta)) / theta3 * np.dot(K, K))

    G = np.empty((4, 4))
    G[0:3, 0:3] = R
    G[0:3, 3] = np.dot(V, v)
    # G[0:3, 3] = np.dot(I-R, K.dot(v)) + np.outer(omega, omega).dot(v) * theta
    G[3, 0:3] = 0
    G[3, 3] = 1
    return G
