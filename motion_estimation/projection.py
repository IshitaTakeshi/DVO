import numpy as np

from motion_estimation.coordinates import compute_pixel_coordinates
from motion_estimation.rigid import transform, log_se3
from motion_estimation.mask import compute_mask

from scipy.ndimage import map_coordinates


def inverse_projection(camera_parameters, pixel_coordinates, depth):
    """
    :math:`S(x)` in the paper

    .. math::
        S(\\mathbf{x}) = \\begin{bmatrix}
            \\frac{(x - o_x) \\cdot h(\\mathbf{x})}{f_x} \\\\
            \\frac{(y - o_y) \\cdot h(\\mathbf{x})}{f_y} \\\\
            h(\\mathbf{x})
        \\end{bmatrix}

    Args:
        camera_parameters (CameraParameters): Camera intrinsic prameters
        pixel_coordinates: pixel_coordinates of shape (n_image_pixels, 2)
        depth: Depth array of shape (n_image_pixels,)
    """

    offset = camera_parameters.offset
    focal_length = camera_parameters.focal_length

    P = pixel_coordinates - offset
    P = (P.T * depth).T
    P = P / focal_length
    return np.vstack((P.T, depth)).T


def projection(camera_parameters, P):
    """
    :math:`\pi(P)` in the paper

    .. math::
        \\pi(P) = \\begin{bmatrix}
            \\frac{X \\cdot f_x}{Z} + o_x \\\\
            \\frac{Y \\cdot f_y}{Z} + o_y \\\\
            h(\\mathbf{x})
        \\end{bmatrix}

    """


    focal_length = camera_parameters.focal_length
    offset = camera_parameters.offset

    def projection_(XY, Z):
        return XY * focal_length / Z + offset

    Q = np.empty((P.shape[0], 2))
    Z = P[:, 2]
    mask = Z > 0

    # the projected coordinates can be calculated properly if the depth is valid
    Q[mask] = projection_(
        P[mask, 0:2],
        Z[mask].reshape(-1, 1)
    )

    # otherwise it is set to nan
    Q[np.logical_not(mask)] = np.nan

    return Q


def reprojection(camera_parameters, depth_map, G):
    # 'reprojection' transforms I0 coordinates to corresponding coordinates in I1

    # 'P' has pixel coordinates in I1 coordinate system, but each pixel
    # coordinate is corresponding to the one in I0

    P = compute_pixel_coordinates(depth_map.shape)

    if np.allclose(log_se3(G), np.zeros(6)):
        # if G is identity, return the identical coordinates
        return P, compute_mask(depth_map, P)

    S = inverse_projection(camera_parameters, P, depth_map.flatten())
    Q = projection(camera_parameters, transform(G, S))

    mask = compute_mask(depth_map, Q)

    # We need to convert 'mask' to the same format as images for convenience

    # Here, mask is in the format below:
    # mask = [
    #     v(x0, y0)  v(x1, y0) ... v(xn, y0)
    #     v(x0, y1)  v(x1, y1) ... v(xn, y1)
    #     ...
    #     v(x0, yj)  v(x1, yj) ... v(xn, yj)
    #     ...
    #     v(x0, ym)  v(x1, ym) ... v(xn, ym)
    # ]
    # where v(x, y) <- {True, False} is a mask at coordinate (x, y)

    mask = mask.reshape(depth_map.shape)

    return Q, mask


def warp(camera_parameters, I1, D0, G):
    # this function samples pixels in I1 and project them to
    # I0 coordinate system

    # 'reprojection' transforms I0 coordinates to
    # the corresponding coordinates in I1

    # 'P' has pixel coordinates in I1 coordinate system, but each pixel
    # coordinate is corresponding to the one in I0
    # Therefore image pixels sampled by 'P' represents I1 transformed into
    # I0 coordinate system

    # 'G' describes the transformation from I0 coordinate system to
    # I1 coordinate system

    P, mask = reprojection(camera_parameters, D0, G)

    # Because 'map_coordinates' requires indices of
    # [row, column] order, the 2nd axis have to be reversed
    # so that it becomes [y, x]

    P = P[:, [1, 0]].T

    # Here,
    # P = [
    #     [y0 y0... y0 ... yi yi... yi ... ym ym ... ym]
    #     [x0 x1... xn ... x0 x1... xn ... x0 x1 ... xn]
    # ]

    # sample pixel sequences from the given image I1
    # warped_image = [
    #     I[y0, x0]  I[y0, x1]  ...  I[y0, xn]
    #     I[y1, x0]  I[y1, x1]  ...  I[y1, xn]
    #     ...
    #     I[ym, x0]  I[ym, x1]  ...  I[ym, xn]
    # ]

    warped_image = map_coordinates(I1, P, mode="constant", cval=np.nan)
    warped_image = warped_image.reshape(D0.shape)

    return warped_image, mask
