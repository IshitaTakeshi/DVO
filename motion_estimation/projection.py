import numpy as np

from motion_estimation.coordinates import compute_pixel_coordinates
from motion_estimation.rigid import transform


def inverse_projection(camera_parameters, pixel_coordinates, depth):
    """
    :math:`S(x)` in the paper

    .. math::
        S(\\mathbf{x}) = \\begin{bmatrix}
            \\frac{(x + o_x) \\cdot h(\\mathbf{x})}{f_x} \\\\
            \\frac{(y + o_y) \\cdot h(\\mathbf{x})}{f_y} \\\\
            h(\\mathbf{x})
        \\end{bmatrix}

    Args:
        camera_parameters (CameraParameters): Camera intrinsic prameters
        pixel_coordinates: pixel_coordinates of shape (n_image_pixels, 2)
        depth: Depth array of shape (n_image_pixels,)
    """

    offset = camera_parameters.offset
    focal_length = camera_parameters.focal_length

    P = pixel_coordinates + offset
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

    G = np.dot(camera_parameters.matrix, G.T)
    G = G[0:2] / G[2]
    return G.T


def reprojection(camera_parameters, depth_map, g):
    pixel_coordinates = compute_pixel_coordinates(depth_map.shape)

    S = inverse_projection(
        camera_parameters,
        pixel_coordinates,
        depth_map.flatten()
    )

    G = transform(g, S)

    P = projection(camera_parameters, G)
    return P
