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
            [0, fy, oy]
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
    return np.dot(g[0:3, 0:3], P) + g[0:3, 3]


def projection(camera_parameters, G):
    """
    :math:`\pi(G)` in the paper

    .. math::
        \\pi(G) = \\begin{bmatrix}
            \\frac{G_1 \\cdot f_x}{G_3} - o_x &
            \\frac{G_2 \\cdot f_y}{G_3} - o_y &
            h(\\mathbf{x})
        \\end{bmatrix}

    """
    focal_length = camera_parameters.focal_length
    offset = camera_parameters.offset
    return (G[0:2] * focal_length + offset) / G[2]


def inverse_projection(camera_parameters, p, depth):
    """
    :math:`S(x)` in the paper

    .. math::
        S(\\mathbf{x}) = \\begin{bmatrix}
            \\frac{(x + o_x) \\cdot h(\\mathbf{x})}{f_x} &
            \\frac{(y + o_y) \\cdot h(\\mathbf{x})}{f_y} &
            h(\\mathbf{x})
        \\end{bmatrix}
    """
    offset = camera_parameters.offset
    focal_length = camera_parameters.focal_length
    s = (p + offset) * depth / focal_length
    return np.concatenate((s, [depth]))


def jacobian_projection(camera_parameters, G):
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
                \\frac{f_{x}}{G_{3}} & 0 & -\\frac{f_{x}}{G_{3}^2} \\\\
                0 & \\frac{f_{y}}{G_{3}} & -\\frac{f_{y}}{G_{3}^2}
            \\end{bmatrix}
        \\end{align}

    """

    fx, fy = camera_parameters.focal_length
    JG = np.array([
        [fx, 0, -G[0] * fx / G[2]],
        [0, fy, -G[1] * fy / G[2]]
    ])
    JG = JG / G[2]
    return JG


def jacobian_3dpoint(P):
    """
    :math:`g(t)` is represented in the vector form :math:`vec(g)`
    In this case we can calculate the multiplication :math:`RP + T` in

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
            \cdot
            vec(g) \\\\
        \\end{align}

    where

    .. math::
        P = [x, y, z]^{\\top}

    """

    I = np.eye(3)
    return np.hstack((I * P[0], I * P[1], I * P[2], I))


def hat3(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def hat6(v):
    return np.array([
        [0, -v[2], v[1], v[3]],
        [v[2], 0, -v[0], v[4]],
        [-v[1], v[0], 0, v[5]],
        [0, 0, 0, 1]
    ])


def jacobian_rigid_motion(g):
    # TODO specify the shape of g
    """
    Args:
        g (np.ndarray): A matrix to represent rotation and translation

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
            T_{13} & T_{23} & T_{33} & T_{41} & T_{42} & T_{43}
        \\end{bmatrix}^{\\top}


    If we represent :math:`g(t)` as

    .. math::
        \\begin{align}
            g(t)
            &= \\begin{bmatrix}
                r_{11} & r_{12} & r_{13} & t_{1} \\\\
                r_{21} & r_{22} & r_{23} & t_{2} \\\\
                r_{31} & r_{32} & r_{33} & t_{3} \\\\
                     0 &      0 &      0 &     1 \\\\
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
                  t_{3} &       0 &   t_{1} & 0 & 1 & 0 \\\\
                  t_{2} &  -t_{1} &       0 & 0 & 0 & 1 \\\\
            \\end{bmatrix}
        \\end{align}
    """

    # left side of Mg
    # ML.shape = (12, 3)
    ML = np.vstack([
        hat3(g[:, 0]),
        hat3(g[:, 1]),
        hat3(g[:, 2]),
        hat3(g[:, 3])
    ])

    # right side
    # MR.shape = (12, 3)
    MR = np.vstack([
        np.zeros((9, 3)),
        np.eye(3)
    ])

    # combine them to form Mg (Mg.shape = (12, 4))
    return np.hstack((ML, MR))


class ImageGradient(object):
    def __init__(self, image):
        self.gradient = self.calc_gradient(image)

    def calc_gradient(self, image):
        dx = image - np.roll(image, -1, axis=0)
        dy = image - np.roll(image, -1, axis=1)
        gradient = np.array([dx, dy])
        gradient = np.swapaxes(gradient, 0, 2)
        gradient = np.swapaxes(gradient, 0, 1)
        print("gradient.shape", gradient.shape)
        return gradient

    def __call__(self, index):
        x, y = index
        return self.gradient[y, x]



class VisualOdometry(object):
    def __init__(self, camera_parameters,
                 reference_image, reference_depth,
                 current_image, current_depth):

        self.reference_image = reference_image
        self.reference_depth = reference_depth
        self.current_image = current_image
        self.current_depth = current_depth

        self.camera_parameters = camera_parameters
        self.image_gradient = ImageGradient(reference_image)
        self.pixel_coordinates =\
            self.compute_pixel_coordinates(reference_image.shape)

    def compute_pixel_coordinates(self, image_shape):
        height, width = image_shape[0:2]
        pixel_coordinates = np.array([(x, y) for x in range(width) for y in range(height)])
        # pixel_coordinates = np.meshgrid(
        #     np.arange(height),
        #     np.arange(width)
        # )
        # pixel_coordinates = np.array(pixel_coordinates)
        return pixel_coordinates

    def compute_jacobian(self, p, g):
        x, y = p
        # print(x, y)
        # print(self.reference_depth.shape)
        depth = self.reference_depth[y, x]
        S = inverse_projection(self.camera_parameters, p, depth)

        G = transform(g, S)

        M = jacobian_rigid_motion(g)
        U = jacobian_3dpoint(G)
        V = jacobian_projection(self.camera_parameters, G)
        W = self.image_gradient(p)
        return W.dot(V).dot(U).dot(M)

    def estimate(self, g):
        J = []
        for p in self.pixel_coordinates:
            J.append(self.compute_jacobian(p, g))
        J = np.vstack(J)
        return J

    def motion_estimation(self, n_coarse_to_fine=5,
                          initial_estimate=None):
        if initial_estimate is None:
            initial_estimate = np.zeros(n_pose_parameters)

        g = np.eye(4)
        xi = initial_estimate  # t0
        y = self.current_image - self.reference_image
        y = y.flatten()
        print("y: ", y)
        print("np.sum(y): ", np.sum(y))
        for i in range(n_coarse_to_fine):
            J = self.estimate(g)
            xi, residuals, rank, singular = np.linalg.lstsq(J, -y)

            print("J.shape: {} y.shape: {}".format(J.shape, y.shape))
            print("xi: ", xi)
            print("residuals: ", residuals)
            g = np.dot(g, hat6(xi))
        return g
