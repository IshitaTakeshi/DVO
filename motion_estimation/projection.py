import numpy as np

from motion_estimation.coordinates import compute_pixel_coordinates
from motion_estimation.rigid import transform, log_se3
from motion_estimation.mask import compute_mask

from scipy.ndimage import map_coordinates


def inverse_projection(camera_parameters, depth_map):
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
        depth_map: Depth map
    """

    offset = camera_parameters.offset
    focal_length = camera_parameters.focal_length

    pixel_coordinates = compute_pixel_coordinates(depth_map.shape)
    depth = depth_map.reshape(-1, 1)
    P = (pixel_coordinates - offset) * depth / focal_length
    return np.hstack((P, depth))


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
    # 'reprojection' transforms I0 coordinates to
    # corresponding coordinates in I1

    # 'C' has pixel coordinates in I1 coordinate system, but each pixel
    # coordinate is corresponding to the one in I0

    if np.allclose(log_se3(G), np.zeros(6)):
        # if G is identity, return the identical coordinates
        C = compute_pixel_coordinates(depth_map.shape)
        return C, compute_mask(depth_map, C)

    S = inverse_projection(camera_parameters, depth_map)
    P = transform(G, S)
    Q = projection(camera_parameters, P)
    return Q, compute_mask(depth_map, Q)


def interpolation(I, Q, order=3):

    # Because 'map_coordinates' requires indices of
    # [row, column] order, the 2nd axis have to be reversed
    # so that it becomes [y, x]

    Q = Q[:, [1, 0]].T

    # Here,
    # Q = [
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

    return map_coordinates(I, Q, mode="constant", cval=np.nan, order=order)


def warp(camera_parameters, I1, D0, G):
    # this function samples pixels in I1 and project them to
    # I0 coordinate system

    # 'reprojection' transforms I0 coordinates to
    # the corresponding coordinates in I1

    # 'Q' has pixel coordinates in I1 coordinate system, but each pixel
    # coordinate is corresponding to the one in I0
    # Therefore image pixels sampled by 'Q' represents I1 transformed into
    # I0 coordinate system

    # 'G' describes the transformation from I0 coordinate system to
    # I1 coordinate system

    Q, mask = reprojection(camera_parameters, D0, G)

    warped_image = interpolation(I1, Q)
    warped_image = warped_image.reshape(D0.shape)

    return warped_image, mask
