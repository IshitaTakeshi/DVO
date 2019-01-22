import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np

from motion_estimation import VisualOdometry, CameraParameters
from motion_estimation.projection import warp
from motion_estimation.quaternion import quaternion_to_rotation
from visualization.plot import plot


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

    def load_color(self, timestamp):
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

    def load_gray(self, timestamp):
        I, D = self.load_color(timestamp)
        return rgb2gray(I), D


def error(image_true, image_pred, mask):
    return np.power(image_true[mask]-image_pred[mask], 2).mean()


def pose_to_matrix(pose):
    R = quaternion_to_rotation(pose[3:])
    t = pose[:3]

    G = np.empty((4, 4))
    G[0:3, 0:3] = R
    G[0:3, 3] = t
    G[3, 0:3] = 0
    G[3, 3] = 1
    return G


def main():
    from visualization.plot import plot
    from visualization.pose import PoseSequenseVisualizer

    np.set_printoptions(suppress=True, precision=8, linewidth=1e8)

    camera_parameters = CameraParameters(
        focal_length=[525.0, 525.0],
        offset=[319.5, 239.5]
    )

    dataset = Dataset()
    timestamps, poses = load_groundtruth()
    start = 0
    end = len(timestamps)
    step = 100
    timestamps, poses = timestamps[start:end:step], poses[start:end:step]

    sequence_true = []
    I0, D0 = dataset.load_gray(timestamps[0])
    G0 = pose_to_matrix(poses[0])
    for timestamp, pose1 in zip(timestamps[1:], poses[1:]):
        I1, D1 = dataset.load_gray(timestamp)
        G1 = pose_to_matrix(pose1)
        DG = np.dot(np.linalg.inv(G0), G1)
        # sequence_true.append(G)

        estimated, mask = warp(camera_parameters, I1, D0, DG)

        if (I1 - I0).sum() > 0:
            print("error(I0, estimated): {}".format(error(I0, estimated, mask)))
            print("error(I1, estimated): {}".format(error(I1, estimated, mask)))
            print("error(I1, I0)       : {}".format(error(I1, I0, mask)))
            print("")

        G0 = G1
        I0, D0 = I1, D1

    exit(0)

    sequence_pred = []
    G = np.eye(4)
    I0, D0 = dataset.load_gray(timestamps[0])
    for timestamp in timestamps[1:]:
        I1, D1 = dataset.load_gray(timestamp)

        vo = VisualOdometry(camera_parameters, I0, D0, I1)
        DG = vo.estimate_motion(n_coarse_to_fine=6)

        print("DG")
        print(DG)

        estimated, mask = warp(camera_parameters, I1, D0, DG)

        print("error(I0, estimated): {}".format(error(I0, estimated, mask)))
        print("error(I1, estimated): {}".format(error(I1, estimated, mask)))
        print("error(I1, I0)       : {}".format(error(I1, I0, mask)))
        if error(I0, estimated, mask) < error(I1, estimated, mask):
            plot(I0, estimated, I1)

        G = np.dot(G, DG)
        sequence_pred.append(G)
        I0, D0 = I1, D1

    # sequence_visualizer = PoseSequenseVisualizer(
    #     (sequence_true, sequence_pred)
    # )
    # sequence_visualizer.show()


main()
