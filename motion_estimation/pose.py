import numpy as np


def rotation_to_quaternion(Q):
    """
    """

    # See https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    xx, xy, xz = Q[0]
    yx, yy, yz = Q[1]
    zx, zy, zz = Q[2]

    diagonal = Q.diagonal()
    trace = diagonal.sum()

    def transform_if_trace_is_nonnegative():
        r = np.sqrt(1 + trace)
        s = 1 / (2 * r)
        w = (1 / 2) * r
        x = (zy - yz) * s
        y = (xz - zx) * s
        z = (yx - xy) * s
        return np.array([w, x, y, z])

    def transform_if_xx_is_largest():
        r = np.sqrt(1 + xx - yy - zz)
        s = 1 / (2 * r)
        w = (zy - yz) * s
        x = (1 / 2) * r
        y = (xy + yx) * s
        z = (zx + xz) * s
        return np.array([w, x, y, z])

    def transform_if_yy_is_largest():
        r = np.sqrt(1 + yy - zz - xx)
        s = 1 / (2 * r)
        w = (xz - zx) * s
        x = (xy + yx) * s
        y = (1 / 2) * r
        z = (yz + zy) * s
        return np.array([w, x, y, z])

    def transform_if_zz_is_largest():
        r = np.sqrt(1 + zz - xx - yy)
        s = 1 / (2 * r)
        w = (yx - xy) * s
        x = (zx + xz) * s
        y = (yz + zy) * s
        z = (1 / 2) * r
        return np.array([w, x, y, z])

    if trace >= 0:
        return transform_if_trace_is_nonnegative()

    argmax = np.argmax(diagonal)

    if argmax == 0:  # Q[0, 0] is the largest diagonal entry
        return transform_if_xx_is_largest()
    if argmax == 1:  # Q[1, 1] is the largest diagonal entry
        return transform_if_yy_is_largest()
    if argmax == 2:  # Q[2, 2] is the largest diagonal entry
        return transform_if_zz_is_largest()


def quaternion_to_rotation(q):
    # See
    # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    # #Quaternion-derived_rotation_matrix

    w, x, y, z = q

    n = np.dot(q, q)
    s = 0 if n == 0 else 2 / n

    v = q[1:]
    wx, wy, wz = s * w * v
    xx, yy, zz = s * v * v

    xy = s * x * y
    yz = s * y * z
    zx = s * z * x

    Q = np.array([
        [1 - (yy + zz), xy - wz, zx + wy],
        [xy + wz, 1 - (xx + zz), yz - wx],
        [zx - wy, yz + wx, 1 - (xx + yy)]
    ])

    return Q
