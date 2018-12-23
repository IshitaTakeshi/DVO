import numpy as np

from skimage.io import imread
from skimage.transform import resize

from motion_estimation.rigid import transformation_matrix
from motion_estimation.coordinates import compute_pixel_coordinates
from motion_estimation.projection import inverse_projection, warp
from motion_estimation.jacobian import (calc_image_gradient,
    jacobian_rigid_motion, jacobian_transform, jacobian_projections)
from motion_estimation.rigid import transform


n_pose_parameters = 6


def compute_weights(r, nu=5, n_iter=10):
    # Kerl Christian, Jürgen Sturm, and Daniel Cremers.
    # "Robust odometry  estimation for RGB-D cameras."
    # Robotics and Automation (ICRA)

    s = np.power(r, 2)

    variance = 1.0
    for i in range(n_iter):
        variance = np.mean(s * (nu + 1) / (nu + s / variance))

    return np.sqrt((nu + 1) / (nu + s / variance));


def compute_jacobian(camera_parameters, image_gradient, depth_map, g):
    """
    Calculate C(x, t)
    """

    pixel_coordinates = compute_pixel_coordinates(depth_map.shape)

    # S.shape = (n_image_pixels, 3)
    S = inverse_projection(
        camera_parameters,
        pixel_coordinates,
        depth_map.flatten()
    )

    # G.shape = (n_image_pixels, 3)
    G = transform(g, S)

    # M.shape = (12, n_pose_parameters)
    # convert xi -> d stack(g)
    M = jacobian_rigid_motion(g)  # M * xi = d(stack(g)) / dt

    # U.shape = (n_image_pixels, 3, 12)
    # convert d stack(g) -> dG
    U = jacobian_transform(S)  # dG / d(stack(g))

    # V.shape = (n_image_pixels, 2, 3)
    # convert dG -> dpi
    V = jacobian_projections(camera_parameters, G)

    # W.shape = (n_image_pixels, 2)
    # convert dpi -> dI
    W = image_gradient

    # J.shape = (n_image_pixels, 3, n_pose_parameters)
    # convert xi -> dG
    J = np.tensordot(U, M, axes=(2, 0))

    # J.shape = (n_image_pixels, 2, n_pose_parameters)
    # convert xi -> dpi
    J = np.einsum('ijk,ikl->ijl', V, J)

    # J.shape = (n_image_pixels, n_pose_parameters)
    J = np.einsum('ij,ijk->ik', W, J)

    return J


class VisualOdometry(object):
    def __init__(self, camera_parameters, I0, D0, I1):
        """
        """

        # TODO check if np.ndim(D0) == np.ndim(I1) == 2

        self.I0 = I0
        self.D0 = D0
        self.I1 = I1

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
                        initial_estimate=None):
        if initial_estimate is None:
            initial_estimate = np.zeros(n_pose_parameters)

        xi = initial_estimate

        for shape in self.image_shapes(n_coarse_to_fine):
            xi = self.estimate_in_layer(
                resize(self.I0, shape),
                resize(self.D0, shape),
                resize(self.I1, shape),
                xi
            )

        return xi

    def estimate_in_layer(self, I0, D0, I1, xi):
        g = transformation_matrix(xi)
        gradient = calc_image_gradient(I1)

        # Transform each pixel of I1 to I0 coordinates

        warped, mask = warp(self.camera_parameters, I1, D0, g)

        J = compute_jacobian(self.camera_parameters, gradient, D0, g)

        assert(mask.shape == warped.shape == I0.shape)

        y = I0.flatten() - warped.flatten()  # comparison on t0 coordinates

        y = y[mask.flatten()]
        J = J[mask.flatten()]

        # NOTE these asserts should be slow
        assert(not np.isnan(y).any())
        assert(not np.isnan(J).any())

        # weights = compute_weights(y)
        # JW = J.T * weights
        # L = np.dot(JW, J)
        # xi = -np.linalg.inv(L).dot(JW).dot(y)
        xi, residuals, rank, singular = np.linalg.lstsq(J, -y, rcond=None)
        return xi
