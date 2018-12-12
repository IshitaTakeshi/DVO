from skimage.io import imread
from skimage.transform import resize
import numpy as np

from motion_estimation.rigid import transformation_matrix
from motion_estimation.coordinates import compute_pixel_coordinates
from motion_estimation.projection import projection, inverse_projection
from motion_estimation.jacobian import (calc_image_gradient,
        jacobian_rigid_motion, jacobian_transform, jacobian_projections)
from motion_estimation.rigid import transform


n_pose_parameters = 6


def compute_mask(camera_parameters, coordinates, image_shape):
    height, width = image_shape[:2]
    coordinates = projection(camera_parameters, coordinates)

    mask_x = np.logical_and(
        0 <= coordinates[:, 0],
        coordinates[:, 0] < width
    )
    mask_y = np.logical_and(
        0 <= coordinates[:, 1],
        coordinates[:, 1] < height
    )
    return np.logical_and(mask_x, mask_y)


def compute_jacobian(camera_parameters, image, depth_map, g):
    # S.shape = (n_image_pixels, 3)


def compute_jacobian(camera_parameters, image_gradient, depth_map, g):
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
    # convert xi -> stack(dg)
    M = jacobian_rigid_motion(g)

    # U.shape = (n_image_pixels, 3, 12)
    # convert stack(dg) -> dG
    U = jacobian_transform(G)

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
    def __init__(self, camera_parameters,
                 reference_image, reference_depth,
                 current_image, current_depth):

        self.reference_image = reference_image
        self.reference_depth = reference_depth
        self.current_image = current_image
        self.current_depth = current_depth

        self.camera_parameters = camera_parameters

    def image_shapes(self, n_coarse_to_fine):
        """
        Generate image shapes for coarse-to-fine
        """

        shape = np.array(self.reference_image.shape)
        shapes = np.array([shape / pow(2, i) for i in range(n_coarse_to_fine)])
        shapes = shapes.astype(np.int64)
        return shapes[::-1]

    def estimate_motion(self, n_coarse_to_fine=5,
                        initial_estimate=None):
        if initial_estimate is None:
            initial_estimate = np.zeros(n_pose_parameters)

        xi = initial_estimate

        for shape in self.image_shapes(n_coarse_to_fine):
            xi = self.estimate_in_layer(
                resize(self.reference_image, shape),
                resize(self.current_image, shape),
                resize(self.reference_depth, shape),
                xi
            )

        return xi

    def estimate_in_layer(self, I0, I1, D0, xi):
        y = I1 - I0
        y = y.flatten()

        g = transformation_matrix(xi)

        gradient = calc_image_gradient(I1)
        J = compute_jacobian(self.camera_parameters, gradient, D0, g)

        xi, residuals, rank, singular = np.linalg.lstsq(J, -y, rcond=None)
        # diff = np.dot(J, xi) + y
        # print(np.power(diff, 2).sum())
        return xi
