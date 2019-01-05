import numpy as np

from motion_estimation.coordinates import compute_pixel_coordinates
from motion_estimation.projection import inverse_projection
from motion_estimation.rigid import transform

# Kerl, Christian.
# "Odometry from rgb-d cameras for autonomous quadrocopters."
# Master's Thesis, Technical University (2012).


def calc_warping_jacobian(camera_parameters, D, G, mask):
    S = inverse_projection(
        camera_parameters,
        compute_pixel_coordinates(D.shape)[mask.flatten()],
        D[mask].flatten()
    )
    P = transform(G, S)
    return calc_jacobian_(camera_parameters, P)


def calc_jacobian_(camera_parameters, P):
    fx, fy = camera_parameters.focal_length

    x, y, z = P[:, 0], P[:, 1], P[:, 2]

    z_squared = np.power(z, 2)
    a = x * y / z_squared

    JW = np.empty((P.shape[0], 2, 6))

    JW[:, 0, 0] = +fx / z
    JW[:, 0, 1] = 0
    JW[:, 0, 2] = -fx * x / z_squared
    JW[:, 0, 3] = -fx * a
    JW[:, 0, 4] = +fx * (1 + x * x / z_squared)
    JW[:, 0, 5] = -fx * y / z

    JW[:, 1, 0] = 0
    JW[:, 1, 1] = +fy / z
    JW[:, 1, 2] = -fy * y / z_squared
    JW[:, 1, 3] = -fy * (1 + y * y / z_squared)
    JW[:, 1, 4] = +fy * a
    JW[:, 1, 5] = +fy * x / z

    return JW


def calc_image_gradient(image):
    """
    Return image gradient `D` of shape (n_image_pixels, 2)
    that :code:`D[y * width + x]` stores the gradient at (x, y)
    """

    dy, dx = np.gradient(image)
    return np.vstack([dx.flatten(), dy.flatten()]).T
