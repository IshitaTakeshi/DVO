import numpy as np
from numpy.linalg import norm


# threshold to switch to the approximated form
EPSILON = 1e-8


def tangent_so3(omega):
    assert(len(omega) == 3)

    return np.array([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0]
    ])


def tangent_se3(xi):
    v, omega = xi[:3], xi[3:]
    K = np.empty((4, 4))
    K[0:3, 0:3] = tangent_so3(omega)
    K[0:3, 3] = v
    K[3] = 0
    return K


def normalize(omega):
    theta = norm(omega)
    if theta == 0:
        return np.zeros(len(omega)), 0
    return omega / theta, theta


def exp_so3(omega):
    I = np.eye(3)

    omega, theta = normalize(omega)

    K = tangent_so3(omega)

    if theta < EPSILON:  # becasue theta = norm(omega) >= 0
        return I + K * theta + np.dot(K, K) * pow(theta, 2) / 2

    return I + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)


def exp_se3(xi):
    v, omega = xi[:3], xi[3:]

    I = np.eye(3)
    R = exp_so3(omega)

    omega, theta = normalize(omega)
    K = tangent_so3(omega)

    if theta < EPSILON:  # since theta = norm(omega) >= 0
        V = I + K * theta / 2 + np.dot(K, K) * pow(theta, 2) / 6
    else:
        V = (I + (1 - np.cos(theta)) / theta * K +
             (theta - np.sin(theta)) / theta * np.dot(K, K))

    G = np.empty((4, 4))
    G[0:3, 0:3] = R
    G[0:3, 3] = np.dot(V, v)
    G[3, 0:3] = 0
    G[3, 3] = 1
    return G


def log_so3(R):
    theta = np.arccos((np.trace(R) - 1) / 2)
    if theta == 0:
        return np.zeros(3), theta

    omega = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ]) / (2 * np.sin(theta))
    return omega, theta


def log_se3(G):
    # Gallier, Jean, and Dianna Xu. "Computing exponentials of skew-symmetric
    # matrices and logarithms of orthogonal matrices." International Journal of
    # Robotics and Automation 18.1 (2003): 10-20.

    R = G[0:3, 0:3]
    t = G[0:3, 3]

    omega, theta = log_so3(R)

    if theta == 0:
        # v == t if theta == 0
        return np.concatenate((t, omega * theta))

    K = tangent_so3(omega)
    I = np.eye(3)
    alpha = -theta / 2
    beta = 1 - theta * np.sin(theta) / (2 * (1 - np.cos(theta)))
    V_inv = I + alpha * K + beta * np.dot(K, K)
    v = V_inv.dot(t)
    return np.concatenate((v, omega * theta))


def rigid_transformation(R, t, P):
    P = np.dot(R, P.T)
    return P.T + t


def transform(G, P):
    """

    .. math::
        G = \\begin{bmatrix}
            R & T \\\\
            \\mathbf{0}^{\\top} & 1 \\\\
        \\end{bmatrix}

    :math:`RP + T`
    """

    return rigid_transformation(G[0:3, 0:3], G[0:3, 3], P)
