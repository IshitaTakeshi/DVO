from pathlib import Path

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np

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


current_image, current_depth = load(0)
camera_parameters = approximate_camera_matrix(current_image.shape)

for i in range(1, 100):
    next_image, next_depth = load(i)

    vo = VisualOdometry(
        camera_parameters,
        current_image, current_depth,
        next_image, next_depth
    )

    current_image, current_image = next_image, next_depth

    motion = vo.estimate_motion()
    print(motion)
