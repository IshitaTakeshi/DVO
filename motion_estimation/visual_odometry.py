from skimage.io import imread
import numpy as np


n_pose_parameters = 6


def to_tuple_if_scalar(value):
    """
    If a scalar value is given, duplicate it and return as a 2 element tuple.
    """
    if isinstance(value, float) or isinstance(value, int):
        return (value, value)
    return value


class CameraParameters(object):
    def __init__(self, focal_length, offset, skew=0):
        ox, oy = to_tuple_if_scalar(offset)
        fx, fy = to_tuple_if_scalar(focal_length)
        s = skew

        self.matrix = np.array([
            [fx, s, ox],
            [0, fy, oy],
            [0, 0, 1]
        ])

        self.focal_length = np.array([fx, fy])
        self.offset = np.array([ox, oy])
        self.skew = s


def transform(g, P):
    """

    .. math::
        g = \\begin{bmatrix}
            R & T \\\\
            \\mathbf{0}^{\\top} & 1 \\\\
        \\end{bmatrix}

    :math:`RP + T`
    """

    P = np.dot(g[0:3, 0:3], P.T)
    return P.T + g[0:3, 3]


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


# @profile
def jacobian_3dpoints(P):
    """
    If :math:`g(t)` is represented in the vector form :math:`vec(g)`,
    we can calculate the multiplication :math:`RP + T` in the form

    .. math::
        \\begin{align}
            RP + T
            &= \\begin{bmatrix}
                xI & yI & zI & I
            \\end{bmatrix}
            \cdot
            vec(g) \\\\
            &= \\begin{bmatrix}
                x & 0 & 0 & y & 0 & 0 & z & 0 & 0 & 1 & 0 & 0 \\\\
                0 & x & 0 & 0 & y & 0 & 0 & z & 0 & 0 & 1 & 0 \\\\
                0 & 0 & x & 0 & 0 & y & 0 & 0 & z & 0 & 0 & 1 \\\\
            \\end{bmatrix}
            \\begin{bmatrix}
                r_{11} \\\\ r_{21} \\\\ r_{31} \\\\
                r_{12} \\\\ r_{22} \\\\ r_{32} \\\\
                r_{13} \\\\ r_{23} \\\\ r_{33} \\\\
                t_{1} \\\\ t_{2} \\\\ t_{3}
            \\end{bmatrix}
        \\end{align}

    where

    .. math::
        P = \\begin{bmatrix} x & y & z \\end{bmatrix}^{\\top}

    """

    n_3dpoints = P.shape[0]
    J = np.zeros((n_3dpoints, 3, 12))
    J[:, 0, 0] = J[:, 1, 1] = J[:, 2, 2] = P[:, 0]
    J[:, 0, 3] = J[:, 1, 4] = J[:, 2, 5] = P[:, 1]
    J[:, 0, 6] = J[:, 1, 7] = J[:, 2, 8] = P[:, 2]
    J[:, [0, 1, 2], [9, 10, 11]] = 1
    return J


def jacobian_projections(camera_parameters, G, epsilon=1e-4):
    """
    Jacobian of the projection function :math:`\pi`

    .. math::
        \\begin{align}
            \\frac{\\partial \\pi(G)}{\\partial G}
            &= \\begin{bmatrix}
                \\frac{\\partial \\pi_{x}}{G_{1}} &
                \\frac{\\partial \\pi_{x}}{G_{2}} &
                \\frac{\\partial \\pi_{x}}{G_{3}} \\\\
                \\frac{\\partial \\pi_{y}}{G_{1}} &
                \\frac{\\partial \\pi_{y}}{G_{2}} &
                \\frac{\\partial \\pi_{y}}{G_{3}}
            \\end{bmatrix} \\\\
            &= \\begin{bmatrix}
                \\frac{f_{x}}{G_{3}} &
                s &
                -\\frac{f_{x} G_{1} + s G_{2}}{G_{3}^2}  \\\\
                0 &
                \\frac{f_{y}}{G_{3}} &
                -\\frac{f_{y} G_{2}}{G_{3}^2}
            \\end{bmatrix}
        \\end{align}

    """

    n_image_pixels = G.shape[0]
    J = np.empty((n_image_pixels, 2, 3))

    Z = G[:, 2]
    Z[Z == 0] = epsilon  # avoid zero divisions

    fx, fy = camera_parameters.focal_length
    s = camera_parameters.skew

    # tile the matrix below `n_image_pixels` times
    # [[fx, s, -(fx*G[0] + s*G[1]) / Z],
    #  [0, fy, -G[1] * fy / Z]]

    J[:, 0, 0] = fx
    J[:, 0, 1] = s
    J[:, 0, 2] = -(fx * G[:, 0] + s * G[:, 1]) / Z
    J[:, 1, 0] = 0
    J[:, 1, 1] = fy
    J[:, 1, 2] = -G[:, 1] * fy / Z
    return J / Z.reshape(Z.shape[0], 1, 1)


def hat3(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def rodrigues(v):
    I = np.eye(3)
    theta = np.linalg.norm(v)

    if np.isclose(theta, 0):
        return I

    v = v / theta
    K = hat3(v)
    cos = np.cos(theta)
    sin = np.sin(theta)
    return I * cos + (1 - sin) * np.outer(v, v) + sin * K


def rigid_transformation(v):
    omega, t = v[:3], v[3:]
    R = rodrigues(omega)
    g = np.empty((4, 4))
    g[0:3, 0:3] = R
    g[0:3, 3] = t
    g[3, 0:3] = 0
    g[3, 3] = 1
    return g


def jacobian_rigid_motion(g):
    # TODO specify the shape of g
    """

    .. math::
        \\frac{dg}{dt} = \\hat{\\xi} \\cdot g(t)

    There exists a matrix :math:`M_g` which satisfies

    .. math::
        stack(\\hat{\\xi} \\cdot g(t)) = M_{g} \\cdot \\xi

    where

    .. math::
        stack(T) =
        \\begin{bmatrix}
            T_{11} & T_{21} & T_{31} & T_{12} & T_{22} & T_{32} &
            T_{13} & T_{23} & T_{33} & T_{14} & T_{24} & T_{34}
        \\end{bmatrix}^{\\top}


    If we represent :math:`g(t)` as

    .. math::
        \\begin{align}
            g(t)
            &= \\begin{bmatrix}
                r_{11} & r_{12} & r_{13} & t_{1} \\\\
                r_{21} & r_{22} & r_{23} & t_{2} \\\\
                r_{31} & r_{32} & r_{33} & t_{3} \\\\
            \\end{bmatrix} \\\\
        \\end{align}

    :math:`M_{g}` becomes

    .. math::
        \\begin{align}
            M_{g}
            &= \\begin{bmatrix}
                      0 &  r_{31} & -r_{21} & 0 & 0 & 0 \\\\
                -r_{31} &       0 &  r_{11} & 0 & 0 & 0 \\\\
                 r_{21} & -r_{11} &       0 & 0 & 0 & 0 \\\\
                      0 &  r_{32} & -r_{22} & 0 & 0 & 0 \\\\
                -r_{32} &       0 &  r_{12} & 0 & 0 & 0 \\\\
                 r_{22} & -r_{12} &       0 & 0 & 0 & 0 \\\\
                      0 &  r_{33} & -r_{23} & 0 & 0 & 0 \\\\
                -r_{33} &       0 &  r_{13} & 0 & 0 & 0 \\\\
                 r_{23} & -r_{13} &       0 & 0 & 0 & 0 \\\\
                      0 &   t_{3} &  -t_{2} & 1 & 0 & 0 \\\\
                 -t_{3} &       0 &   t_{1} & 0 & 1 & 0 \\\\
                  t_{2} &  -t_{1} &       0 & 0 & 0 & 1
            \\end{bmatrix}
        \\end{align}
    """

    # left side of Mg
    # ML.shape = (12, 3)
    ML = np.vstack([
        hat3(-g[:, 0]),
        hat3(-g[:, 1]),
        hat3(-g[:, 2]),
        hat3(-g[:, 3])
    ])

    # right side
    # MR.shape = (12, 3)
    MR = np.vstack([
        np.zeros((9, 3)),
        np.eye(3)
    ])

    # combine them to form Mg (Mg.shape = (12, 4))
    return np.hstack((ML, MR))


def calc_image_gradient(image):
    """
    Return image gradient `D` of shape (n_image_pixels, 2)
    that :code:`D[y * width + y]` stores the gradient at (x, y)
    """

    dx = image - np.roll(image, -1, axis=0)
    dy = image - np.roll(image, -1, axis=1)
    return np.vstack([dx.flatten(), dy.flatten()]).T


class VisualOdometry(object):
    def __init__(self, camera_parameters,
                 reference_image, reference_depth,
                 current_image, current_depth):

        self.reference_image = reference_image
        self.reference_depth = reference_depth
        self.current_image = current_image
        self.current_depth = current_depth

        self.camera_parameters = camera_parameters
        self.image_gradient = calc_image_gradient(reference_image)

    def compute_pixel_coordinates(self, image_shape):
        height, width = image_shape[0:2]
        pixel_coordinates = np.array([(x, y) for x in range(width) for y in range(height)])
        # pixel_coordinates = np.meshgrid(
        #     np.arange(height),
        #     np.arange(width)
        # )
        # pixel_coordinates = np.array(pixel_coordinates)
        return pixel_coordinates

    # @profile
    def compute_jacobian(self, depth_map, g):
        # S.shape = (n_image_pixels, 3)

        pixel_coordinates = self.compute_pixel_coordinates(depth_map.shape)

        S = inverse_projection(
            self.camera_parameters,
            pixel_coordinates,
            depth_map.flatten()
        )

        # G.shape = (n_image_pixels, 3)
        G = transform(g, S)

        # M.shape = (12, n_pose_parameters)
        M = jacobian_rigid_motion(g)
        # U.shape = (n_3dpoints, 3, 12)
        U = jacobian_3dpoints(G)
        # V.shape = (n_3dpoints, 2, 3)
        V = jacobian_projections(self.camera_parameters, G)

        # Equivalent to J = np.einsum('kli,ij->klj', U, M)
        # J.shape = (n_3dpoints, 3, n_pose_parameters)
        J = np.tensordot(U, M, axes=(2, 0))

        # Equivalent to J = np.einsum('ilj,ijk->lk', V, J)
        # J.shape = (2, n_pose_parameters)
        J = np.tensordot(V, J, axes=((0, 2), (0, 1)))

        # W.shape = (n_image_pixels, 2)
        W = self.image_gradient
        return np.dot(W, J)  # (n_image_pixels, n_pose_parameters)

    def estimate_motion(self, n_coarse_to_fine=5,
                        initial_estimate=None):
        if initial_estimate is None:
            initial_estimate = np.zeros(n_pose_parameters)

        xi = initial_estimate  # t0

        for i in range(n_coarse_to_fine):
            xi = self.estimate_in_layer(
                self.reference_image,
                self.current_image,
                self.reference_depth,
                xi
            )
        return xi

    # @profile
    def estimate_in_layer(self, I0, I1, D0, xi):
        y = I1 - I0
        y = y.flatten()

        J = self.compute_jacobian(D0, rigid_transformation(xi))

        xi, residuals, rank, singular = np.linalg.lstsq(J, -y, rcond=None)
        return xi
