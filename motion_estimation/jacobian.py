import numpy as np

# Kerl, Christian.
# "Odometry from rgb-d cameras for autonomous quadrocopters."
# Master's Thesis, Technical University (2012).

def jacobian_transform(P):
    """
    If :math:`g(t)` is represented in the vector form :math:`vec(g)`,
    we can calculate the multiplication :math:`RP + T` in the form

    .. math::
        \\begin{align}
            RP + T
            \cdot
            vec(g) \\\\
            &= \\begin{bmatrix}
                x & y & z & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
                0 & 0 & 0 & 0 & x & y & z & 1 & 0 & 0 & 0 & 0 \\\\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & x & y & z & 1 \\\\
            \\end{bmatrix}
            \\begin{bmatrix}
                r_{11} \\\\ r_{12} \\\\ r_{13} \\\\ t_{1} \\\\
                r_{21} \\\\ r_{22} \\\\ r_{23} \\\\ t_{2} \\\\
                r_{31} \\\\ r_{32} \\\\ r_{33} \\\\ t_{3}
            \\end{bmatrix}
        \\end{align}

    where

    .. math::
        P = \\begin{bmatrix} x & y & z \\end{bmatrix}^{\\top}

    Args:
        P (np.ndarray): Array of shape (n_3dpoints, 3)
    Returns:
        np.ndarray:
            Jacobian :code:`J` of shape (n_image_pixels, 3, 12)
            which :code:`J[i]` represents :math:`dG(P[i])/dg`
    """

    n_3dpoints = P.shape[0]
    J = np.zeros((n_3dpoints, 3, 12))
    J[:, 0, 0:3] = J[:, 1, 4:7] = J[:, 2, 8:11] = P
    J[:, [0, 1, 2], [3, 7, 11]] = 1
    return J


def jacobian_rigid_motion(g):
    # TODO specify the shape of g
    """
    Returns `Mg` in the paper

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
                      0 &  r_{32} & -r_{22} & 0 & 0 & 0 \\\\
                      0 &  r_{33} & -r_{23} & 0 & 0 & 0 \\\\
                      0 &   t_{3} &  -t_{2} & 1 & 0 & 0 \\\\
                -r_{31} &       0 &  r_{11} & 0 & 0 & 0 \\\\
                -r_{32} &       0 &  r_{12} & 0 & 0 & 0 \\\\
                -r_{33} &       0 &  r_{13} & 0 & 0 & 0 \\\\
                 -t_{3} &       0 &   t_{1} & 0 & 1 & 0 \\\\
                 r_{21} & -r_{11} &       0 & 0 & 0 & 0 \\\\
                 r_{22} & -r_{12} &       0 & 0 & 0 & 0 \\\\
                 r_{23} & -r_{13} &       0 & 0 & 0 & 0 \\\\
                  t_{2} &  -t_{1} &       0 & 0 & 0 & 1
            \\end{bmatrix}
        \\end{align}

    Args:
        g (np.ndarray): A matrix to represent rotation and translation

    """

    M = np.zeros((12, 6))
    M[0:4, 1] = g[2]
    M[0:4, 2] = -g[1]

    M[4:8, 0] = -g[2]
    M[4:8, 2] = g[0]

    M[8:12, 0] = g[1]
    M[8:12, 1] = -g[0]

    M[[3, 7, 11], [3, 4, 5]] = 1
    return M


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
                0 &
                -\\frac{f_{x} G_{1}}{G_{3}^2}  \\\\
                0 &
                \\frac{f_{y}}{G_{3}} &
                -\\frac{f_{y} G_{2}}{G_{3}^2}
            \\end{bmatrix}
        \\end{align}

    Args:
        G (np.ndarray): n_image_pixels
    """

    n_image_pixels = G.shape[0]
    J = np.empty((n_image_pixels, 2, 3))

    Z = G[:, 2]
    Z[Z == 0] = epsilon  # avoid zero divisions

    fx, fy = camera_parameters.focal_length

    # J[i] = [
    #   [fx, 0, -fx*G[i, 0] / Z[i]],
    #   [0, fy, -G[i, 1] * fy / Z[i]]
    # ] / Z[i]

    J[:, 0, 0] = fx
    J[:, 0, 1] = 0
    J[:, 0, 2] = -fx * G[:, 0] / Z
    J[:, 1, 0] = 0
    J[:, 1, 1] = fy
    J[:, 1, 2] = -fy * G[:, 1] / Z
    J = J / Z.reshape(Z.shape[0], 1, 1)
    return J


def calc_image_gradient(image):
    """
    Return image gradient `D` of shape (n_image_pixels, 2)
    that :code:`D[y * width + x]` stores the gradient at (x, y)
    """

    dy, dx = np.gradient(image)
    return np.vstack([dx.flatten(), dy.flatten()]).T
