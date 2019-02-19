import numpy as np

from skimage.io import imread
from skimage.transform import resize

from motion_estimation.rigid import exp_se3
from motion_estimation.projection import warp
from motion_estimation.jacobian import calc_image_gradient, calc_warping_jacobian
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


class VisualOdometry(object):
    def __init__(self, camera_parameters, I0, D0, I1,
                 epsilon=5e-4, max_iter=200):
        """
        """

        # TODO check if np.ndim(D0) == np.ndim(I1) == 2

        self.I0 = I0
        self.D0 = D0
        self.I1 = I1

        self.epsilon = epsilon
        self.max_iter = max_iter

        self.camera_parameters = camera_parameters

    def image_shapes(self, n_coarse_to_fine):
        """
        Generate image shapes for coarse-to-fine
        """

        shape = np.array(self.I0.shape)
        shapes = np.array([shape / pow(2, i) for i in range(n_coarse_to_fine)])
        shapes = shapes.astype(np.int64)
        # TODO raise ValueError
        # if shape[0] * shape[1] < 6 (number of pose parameters)
        return shapes[::-1]

    def estimate_motion(self, n_coarse_to_fine=5,
                        initial_pose=np.zeros(6)):
        """Estimate a motion from t1 to t0"""

        G = exp_se3(initial_pose)
        for shape in self.image_shapes(n_coarse_to_fine)[:-2]:
            print("shape: {}".format(shape))
            try:
                G = self.estimate_in_layer(
                    resize(self.I0, shape),
                    resize(self.D0, shape),
                    resize(self.I1, shape),
                    G
                )
            except np.linalg.linalg.LinAlgError as e:
                return exp_se3(initial_pose)
        return G

    def estimate_in_layer(self, I0, D0, I1, G):
        previous_error = np.inf

        for k in range(self.max_iter):
            DG, current_error = self.calc_pose_update(I0, D0, I1, G)
            print("k: {:>3d}  error: {:>3.3f}".format(k, current_error))
            print("DG")
            print(DG)

            if current_error < self.epsilon:
                break

            if abs(current_error - previous_error) < current_error * 0.01:
                break

            # if current_error > previous_error:
            #     break

            # if abs(current_error - previous_error) < self.epsilon:
            #     break

            G = G.dot(DG)

            previous_error = current_error
        return G

    def calc_pose_update(self, I0, D0, I1, G, min_depth=1e-8):
        # warp from t0 to t1
        # 'warped' represents I0 transformed onto the t1 coordinates
        warped, mask = warp(self.camera_parameters, I1, D0, G)

        r = -(warped[mask] - I0[mask]).flatten()

        # image_gradient.shape == (n_image_pixels, 2)
        image_gradient = calc_image_gradient(I0)[mask.flatten()]

        from motion_estimation.jacobian import calc_jacobian
        from motion_estimation.coordinates import compute_pixel_coordinates
        from motion_estimation.projection import inverse_projection

        P = inverse_projection(
            self.camera_parameters,
            compute_pixel_coordinates(D0.shape)[mask.flatten()],
            D0[mask].flatten()
        )
        JW = calc_jacobian(self.camera_parameters, P)

        # JW.shape == (n_image_pixels, 2, 6)
        # JW = calc_warping_jacobian(self.camera_parameters, D0, G, mask)

        # J.shape == (n_image_pixels, 6)
        J = np.einsum('ij,ijk->ik', image_gradient, JW)

        # weights = compute_weights_tukey(r)
        # weights = compute_weights_student_t(r)
        xi = solve_linear_equation(J, r, weights=None)
        DG = exp_se3(xi)

        error = calc_error(r, weights=None)

        return DG, error
