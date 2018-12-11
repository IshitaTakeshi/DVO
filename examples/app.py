from pathlib import Path

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

from skimage.io import imread
from skimage.color import rgb2gray
from scipy.ndimage import map_coordinates
from matplotlib import pyplot as plt

from motion_estimation import VisualOdometry, CameraParameters

image_root = Path("dataset", "20fps_images_archieve")
depth_root = Path("dataset", "20fps_real_GT_archieve")


def load(frame):
    prefix = "scene_09_{:04}".format(frame)

    path = Path(image_root, prefix + ".png")
    image = imread(str(path))
    image = rgb2gray(image)

    path = Path(depth_root, prefix + ".depth")
    depth = np.loadtxt(str(path))
    depth = depth.reshape(image.shape[:2])

    return image, depth


def approximate_camera_matrix(image_shape):
    H, W = image_shape[:2]
    return CameraParameters(focal_length=W, offset=[W/2, H/2])


from motion_estimation.visual_odometry import (
    compute_pixel_coordinates, inverse_projection,
    rigid_transformation, transform, projection)


def reprojection(camera_parameters, depth_map, xi):
    pixel_coordinates = compute_pixel_coordinates(depth_map.shape)

    S = inverse_projection(
        camera_parameters,
        pixel_coordinates,
        depth_map.flatten()
    )

    g = rigid_transformation(-xi)
    print(g)
    G = transform(g, S)

    P = projection(camera_parameters, G)
    return P


def warp(camera_parameters, image, depth, xi):
    P = reprojection(camera_parameters, depth, xi)
    # P = compute_pixel_coordinates(depth.shape)
    P = P[:, [1, 0]]
    warped = map_coordinates(image, P.T)
    warped = warped.reshape(image.shape[1], image.shape[0]).T
    return warped


def plot(image1, image2):
    fig = plt.figure()

    ax = fig.add_subplot(211)
    ax.imshow(image1)

    ax = fig.add_subplot(212)
    ax.imshow(image2)

    plt.show()




camera_parameters = CameraParameters(focal_length=1280, offset=0)

current_image, current_depth = load(1)
next_image, next_depth = load(2)

vo = VisualOdometry(
    camera_parameters,
    current_image, current_depth,
    next_image, next_depth
)
motion = vo.estimate_motion()

estimated_image = warp(camera_parameters,
                       current_image, current_depth, motion)

plot(current_image, estimated_image)
current_image, current_image = next_image, next_depth

print(motion)
