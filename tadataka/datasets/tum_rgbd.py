from collections import namedtuple
import csv
from pathlib import Path

from skimage.color import rgb2gray
from skimage.io import imread
import numpy as np

from tadataka.quaternion import rotation_to_quaternion, quaternion_to_rotation
from tadataka.datasets.fileio import decomment

Frame = namedtuple(
    "Frame",
    [
        "timestamp_rgb",
        "timestamp_depth",
        "image",
        "depth_map"
    ]
)


def matrix_to_pose(G):
    R = G[0:3, 0:3]
    t = G[0:3, 3]
    q = rotation_to_quaternion(R)
    return np.concatenate((t, q))


def pose_to_matrix(pose):
    x, y, z, w = pose[3:]  # because of the TUM dataset format
    R = quaternion_to_rotation(np.array([w, x, y, z]))
    t = pose[:3]

    G = np.empty((4, 4))
    G[0:3, 0:3] = R
    G[0:3, 3] = t
    G[3, 0:3] = 0
    G[3, 3] = 1
    return G


def load_pose_sequence(path):
    timestamps = []
    poses = []
    with open(str(path), "r") as f:
        reader = csv.reader(decomment(f), delimiter=' ')

        for row in reader:
            timestamps.append(row[0])
            poses.append([float(e) for e in row[1:]])
    return PoseSequence(timestamps, poses)


def sort_by_key(tuples):
    return sorted(tuples, key=lambda x: x[0])


# TODO add tests
class PoseSequence(object):
    def __init__(self, timestamps=[], poses=[]):
        self.pose_sequence = {}
        for timestamp, pose in zip(timestamps, poses):
            self.pose_sequence[timestamp] = pose

    def add(self, timestamp, G):
        self.pose_sequence[timestamp] = matrix_to_pose(G)

    def save(self, filename):
        pose_sequence = sort_by_key(self.pose_sequence.items())
        with open(filename, "w") as f:
            writer = csv.writer(f, delimiter=' ')
            for timestamp, pose in pose_sequence:
                writer.writerow([timestamp, *pose])


class TUMDataset(object):
    def __init__(self, dataset_root, depth_factor=5000):
        self.dataset_root = dataset_root

        # note that each timestamp is represented in string
        timestamps_rgb, paths_rgb, timestamps_depth, paths_depth = self.init()
        self.timestamps_rgb = timestamps_rgb
        self.paths_rgb = paths_rgb
        self.timestamps_depth = timestamps_depth
        self.paths_depth = paths_depth
        self.depth_factor = depth_factor

    def init(self):
        path = Path(self.dataset_root, "rgbd.txt")
        with open(str(path), "r") as f:
            reader = csv.reader(f, delimiter=' ')

            timestamps_rgb = []
            paths_rgb = []
            timestamps_depth = []
            paths_depth = []
            for row in reader:
                timestamps_rgb.append(row[0])
                paths_rgb.append(row[1])
                timestamps_depth.append(row[2])
                paths_depth.append(row[3])
        return timestamps_rgb, paths_rgb, timestamps_depth, paths_depth

    @property
    def size(self):
        return len(self.timestamps_rgb)

    def load_color(self, index):
        timestamp_rgb = self.timestamps_rgb[index]
        timestamp_depth = self.timestamps_depth[index]

        path_rgb = str(Path(self.dataset_root, self.paths_rgb[index]))
        path_depth = str(Path(self.dataset_root, self.paths_depth[index]))

        I = imread(path_rgb)
        D = imread(path_depth)
        D = D / self.depth_factor

        return Frame(timestamp_rgb, timestamp_depth, I, D)

    def load_gray(self, index):
        frame = self.load_color(index)
        # replace the image with the gray one
        return frame._replace(image=rgb2gray(frame.image))
