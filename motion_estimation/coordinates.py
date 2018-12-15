import numpy as np


def compute_pixel_coordinates(image_shape):
    """

    Example:
	For array of shape (3, 2) (if it was an image, height=3 and width=2),

	>>> coordinates.compute_pixel_coordinates((3, 2))
	array([[0, 0],
	       [1, 0],
	       [0, 1],
	       [1, 1],
	       [0, 2],
	       [1, 2]])

    """

    height, width = image_shape[0:2]

    xs, ys = np.meshgrid(
        np.arange(width),
        np.arange(height)
    )

    P = np.vstack((xs.flatten(), ys.flatten()))
    return P.T
