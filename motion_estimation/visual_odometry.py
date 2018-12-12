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


def compute_jacobian(camera_parameters, image, depth_map, g):
    # S.shape = (n_image_pixels, 3)

    image_gradient = calc_image_gradient(image)

    pixel_coordinates = compute_pixel_coordinates(depth_map.shape)

    S = inverse_projection(
        camera_parameters,
        pixel_coordinates,
        depth_map.flatten()
    )

    # G.shape = (n_image_pixels, 3)
    G = transform(g, S)

    # M.shape = (12, n_pose_parameters)
    M = jacobian_rigid_motion(g)
    # U.shape = (n_3dpoints, 3, 12)
    U = jacobian_transform(G)
    # V.shape = (n_3dpoints, 2, 3)
    V = jacobian_projections(camera_parameters, G)

    # Equivalent to J = np.einsum('kli,ij->klj', U, M)
    # J.shape = (n_3dpoints, 3, n_pose_parameters)
    J = np.tensordot(U, M, axes=(2, 0))

    # Equivalent to J = np.einsum('ilj,ijk->lk', V, J)
    # J.shape = (2, n_pose_parameters)
    J = np.tensordot(V, J, axes=((0, 2), (0, 1)))

    # W.shape = (n_image_pixels, 2)
    W = image_gradient

    J = np.dot(W, J)  # (n_image_pixels, n_pose_parameters)
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
        return shapes.astype(np.int64)

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
        # print(I0.shape, I1.shape, D0.shape)
        y = I1 - I0
        y = y.flatten()

        g = transformation_matrix(xi)
        J = compute_jacobian(self.camera_parameters, I0, D0, g)
        print("power(J, 2).sum()", np.power(J, 2).sum())

        xi, residuals, rank, singular = np.linalg.lstsq(J, -y, rcond=None)
        # diff = np.dot(J, xi) + y
        # print(np.power(diff, 2).sum())
        return xi
