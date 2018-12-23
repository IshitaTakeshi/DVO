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
from motion_estimation.projection import reprojection, warp
from motion_estimation.rigid import transformation_matrix
from visualization.plot import plot


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


camera_parameters = CameraParameters(focal_length=10, offset=0)

current_image, current_depth = load(1)
next_image, next_depth = load(3)
current_depth = current_depth
next_depth = next_depth
vo = VisualOdometry(camera_parameters,
                    current_image, current_depth, next_image)
motion = vo.estimate_motion(n_coarse_to_fine=8)

print("motion")
print(transformation_matrix(motion))

g = transformation_matrix(-motion)
estimated_image, mask = warp(camera_parameters, current_image, current_depth, g)
estimated_image[np.logical_not(mask)] = 0.0

plot(current_image, estimated_image, next_image)
