import numpy as np

def compute_pixel_coordinates(image_shape):
    # TODO accelerate

    height, width = image_shape[0:2]
    pixel_coordinates = np.array(
        [(x, y) for x in range(width) for y in range(height)]
    )
    # pixel_coordinates = np.meshgrid(
    #     np.arange(height),
    #     np.arange(width)
    # )
    # pixel_coordinates = np.array(pixel_coordinates)
    return pixel_coordinates

