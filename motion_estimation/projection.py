import numpy as np


def inverse_projection(camera_parameters, P, depth):
    """
    :math:`S(x)` in the paper

    .. math::
        S(\\mathbf{x}) = \\begin{bmatrix}
            \\frac{(x + o_x) \\cdot h(\\mathbf{x})}{f_x} \\\\
            \\frac{(y + o_y) \\cdot h(\\mathbf{x})}{f_y} \\\\
            h(\\mathbf{x})
        \\end{bmatrix}
    """

    offset = camera_parameters.offset
    focal_length = camera_parameters.focal_length

    P = P + offset
    P = (P.T * depth).T
    P = P / focal_length
    return np.vstack((P.T, depth)).T


def projection(camera_parameters, G):
    """
    :math:`\pi(G)` in the paper

    .. math::
        \\pi(G) = \\begin{bmatrix}
            \\frac{G_1 \\cdot f_x}{G_3} - o_x \\\\
            \\frac{G_2 \\cdot f_y}{G_3} - o_y \\\\
            h(\\mathbf{x})
        \\end{bmatrix}

    """

    Z = np.dot(camera_parameters.matrix, G.T)
    Z = Z[0:2] / Z[2]
    return Z.T
