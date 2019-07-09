import numpy as np


def compute_weights_student_t(r, nu=5, n_iter=10):
    # Kerl Christian, Jürgen Sturm, and Daniel Cremers.
    # "Robust odometry estimation for RGB-D cameras."
    # Robotics and Automation (ICRA)

    s = np.power(r, 2)

    variance = 1.0
    for i in range(n_iter):
        variance = np.mean(s * (nu + 1) / (nu + s / variance))

    return np.sqrt((nu + 1) / (nu + s / variance))


def tukey(x, beta):
    w = np.zeros(x.shape)
    mask = np.abs(x) <= beta
    w[mask] = np.power(1 - np.power(x[mask] / beta, 2), 2)
    return w


def median_absolute_deviation(x):
    return np.median(np.abs(x - np.median(x)))


def compute_weights_tukey(r, beta=4.6851, c=1.4826):
    # Equation 4.28 in the paper
    sigma_mad = c * median_absolute_deviation(r)
    return tukey(r / sigma_mad, beta)
