from pathlib import Path

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

from skimage.io import imread
from skimage.color import rgb2gray
from scipy.ndimage import map_coordinates

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
    return image, depth / 50000.0

camera_parameters = CameraParameters(focal_length=10, offset=0)

I0, D0 = load(1)
I1, D1 = load(3)

vo = VisualOdometry(camera_parameters, I0, D0, I1)
motion = vo.estimate_motion(n_coarse_to_fine=8)

print("motion")
print(g)

warped, mask = warp(camera_parameters, I1, D0, g)
warped[np.logical_not(mask)] = 0.0
plot(I0, warped, I1)
