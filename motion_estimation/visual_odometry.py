import numpy as np
from numpy.linalg import norm

from skimage.io import imread
from skimage.transform import resize

from motion_estimation.camera import CameraParameters
from motion_estimation.rigid import exp_se3, log_se3, transform
from motion_estimation.coordinates import compute_pixel_coordinates
from motion_estimation.projection import warp, inverse_projection
from motion_estimation.jacobian import calc_image_gradient, calc_jacobian
from motion_estimation.weights import (compute_weights_tukey,
                                       compute_weights_student_t)


n_pose_parameters = 6


def calc_error(r, weights=None):
    if weights is None:
        return np.dot(r, r)
    return np.dot(r * weights, r)  # r * W * r


def solve_linear_equation(J, r, weights=None):
    # (J^T * W * J)^{-1} * J^T * W * r
    if weights is None:
        JW = J.T
    else:
        JW = J.T * weights
    L = JW.dot(J)  # L = J.T * W * J
    K = np.linalg.inv(L)
    return K.dot(JW).dot(r)


def level_to_ratio(level):
    return 1 / pow(2, level)


def calc_pose_update(camera_parameters, I0, D0, I1, G, min_depth=1e-8):
    S = inverse_projection(
        camera_parameters,
        compute_pixel_coordinates(D0.shape),
        D0.flatten()
    )
    P = transform(G, S)  # to the t1 coordinates
    # mask = P[:, 2] > 0

    DX, DY = calc_image_gradient(I1)

    # Transform onto the t0 coordinate
    # means that
    # 1. backproject each pixel in the t0 frame to 3D
    # 2. transform the 3D points to t1 coordinates
    # 3. reproject the transformed 3D points to the t1 coordinates
    # 4. interpolate image gradient maps using the reprojected coordinates

    dx_warped, mask = warp(camera_parameters, DX, D0, G)
    dy_warped, mask = warp(camera_parameters, DY, D0, G)

    # J.shape == (n_image_pixels, 6)
    J = calc_jacobian(
        camera_parameters,
        dx_warped[mask].flatten(),  # (n_available_pixels,)
        dy_warped[mask].flatten(),  # (n_available_pixels,)
        P[mask.flatten(), :]        # (n_available_pixels, 3)
    )

    warped, _ = warp(camera_parameters, I1, D0, G)
    r = -(warped[mask] - I0[mask]).flatten()

    # weights = compute_weights_tukey(r)
    # weights = compute_weights_student_t(r)
    xi = solve_linear_equation(J, r, weights=None)
    DG = exp_se3(xi)

    error = calc_error(r, weights=None)

    return DG, error


class VisualOdometry(object):
    def __init__(self, camera_parameters, I0, D0, I1,
                 epsilon=1e-3, max_iter=200):
        """
        """

        # TODO check if np.ndim(D0) == np.ndim(I1) == 2

        self.I0 = I0
        self.D0 = D0
        self.I1 = I1

        self.epsilon = epsilon
        self.max_iter = max_iter

        self.camera_parameters = camera_parameters

    def estimate_motion(self, n_coarse_to_fine=5,
                        initial_pose=np.zeros(6)):
        """Estimate a motion from t1 to t0"""

        levels = list(reversed(range(n_coarse_to_fine)))

        G = exp_se3(initial_pose)
        for level in levels[:-2]:
            print("\n")
            print("level: {}".format(level))
            try:
                G = self.estimate_motion_at(level, G)
            except np.linalg.linalg.LinAlgError as e:
                print(e)
                return exp_se3(initial_pose)
        return G

    def camera_parameters_at(self, level):
        focal_length = self.camera_parameters.focal_length
        offset = self.camera_parameters.offset
        ratio = level_to_ratio(level)
        return CameraParameters(focal_length * ratio, offset * ratio)

    def image_shape_at(self, level):
        shape = np.array(self.I0.shape)
        ratio = level_to_ratio(level)
        return shape * ratio

    def estimate_motion_at(self, level, G):
        shape = self.image_shape_at(level)
        I0 = resize(self.I0, shape)
        D0 = resize(self.D0, shape)
        I1 = resize(self.I1, shape)

        for k in range(self.max_iter):
            DG, error = calc_pose_update(
                self.camera_parameters_at(level),
                I0, D0, I1, G
            )

            xi = log_se3(DG)
            print("k: {:>4d} norm(xi): {:4.3f}  error: {:4.3f}  xi: {}".format(
                  k, norm(xi), error, xi))

            if norm(xi) < self.epsilon:
                break

            G = G.dot(DG)
        return G
