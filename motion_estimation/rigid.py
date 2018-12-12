import numpy as np
from motion_estimation.rotation import rodrigues


def transform(g, P):
    """

    .. math::
        g = \\begin{bmatrix}
            R & T \\\\
            \\mathbf{0}^{\\top} & 1 \\\\
        \\end{bmatrix}

    :math:`RP + T`
    """

    P = np.dot(g[0:3, 0:3], P.T)
    return P.T + g[0:3, 3]


def transformation_matrix(v):
    omega, t = v[:3], v[3:]
    R = rodrigues(omega)
    g = np.empty((4, 4))
    g[0:3, 0:3] = R
    g[0:3, 3] = t
    g[3, 0:3] = 0
    g[3, 3] = 1
    return g
