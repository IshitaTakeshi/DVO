import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np

from motion_estimation import VisualOdometry, CameraParameters


# dataset format is explained at
# https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats#
# intrinsic_camera_calibration_of_the_kinect

dataset_root = Path("dataset", "rgbd_dataset_freiburg1_desk")
depth_factor = 5000


def is_comment(row):
    return row[0] == "#"


def load_groundtruth():
    path = str(Path(dataset_root, "groundtruth.txt"))
    data = np.loadtxt(path)
    timestamps = data[:, 0]
    poses = data[:, 1:]
    return timestamps, poses


class Dataset(object):
    def __init__(self):
        self.rgb_root = Path(dataset_root, "rgb")
        self.depth_root = Path(dataset_root, "depth")
        self.rgb_timestamps = self.load_timestamps(self.rgb_root)
        self.depth_timestamps = self.load_timestamps(self.depth_root)

    def filename_to_timestamp(self, filename):
        return float(filename.replace(".png", ""))

    def timestamp_to_filename(self, timestamp):
        return "{:.6f}.png".format(timestamp)

    def load_timestamps(self, directory):
        filenames = [p.name for p in directory.glob("*.png")]
        timestamps = [self.filename_to_timestamp(f) for f in filenames]
        return np.sort(timestamps)

    def search_nearest_timestamp(self, timestamps, query):
        index = np.searchsorted(timestamps, query)
        return timestamps[index]

    def load_nearest(self, root, timestamps, query):
        nearest = self.search_nearest_timestamp(timestamps, query)
        filename = self.timestamp_to_filename(nearest)
        path = str(Path(root, filename))
        return imread(path)

    def load(self, timestamp):
        rgb_image = self.load_nearest(
            self.rgb_root,
            self.rgb_timestamps,
            timestamp
        )
        depth_image = self.load_nearest(
            self.depth_root,
            self.depth_timestamps,
            timestamp
        )
        return rgb_image, depth_image / depth_factor


def main():
    timestamps, poses = load_groundtruth()

    dataset = Dataset()
    camera_parameters = CameraParameters(
        focal_length=[525.0, 525.0],
        offset=[319.5, 239.5]
    )


    I0, D0 = dataset.load(timestamps[1200])
    I0 = rgb2gray(I0)

    I1, D1 = dataset.load(timestamps[1244])
    I1 = rgb2gray(I1)

    print("Images are same: {}".format((I0 == I1).all()))

    vo = VisualOdometry(camera_parameters, I0, D0, I1)

    # I0, D0 = I1, D1

    motion = vo.estimate_motion(n_coarse_to_fine=1)
    print(motion)

main()
