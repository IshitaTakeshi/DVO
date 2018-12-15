import numpy as np


def is_in_rage(image_shape, coordinates):
    height, width = image_shape[:2]

    mask_x = np.logical_and(
        0 <= coordinates[:, 0],
        coordinates[:, 0] <= width-1
    )

    mask_y = np.logical_and(
        0 <= coordinates[:, 1],
        coordinates[:, 1] <= height-1
    )

    return np.logical_and(mask_x, mask_y)


def compute_mask(depth_map, pixel_coordinates):
    # depth_map and pixel_coordinates has to be in the same coordinate system

    depth_mask = depth_map > 0
    range_mask = is_in_rage(depth_map.shape, pixel_coordinates)
    range_mask = range_mask.reshape(depth_map.shape)
    return np.logical_and(depth_mask, range_mask)
