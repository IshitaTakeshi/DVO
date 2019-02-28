from collections import namedtuple
import csv
from pathlib import Path

from skimage.color import rgb2gray
from skimage.io import imread

from tadataka.quaternion import rotation_to_quaternion


Frame = namedtuple(
    "Frame",
    [
        "timestamp_rgb",
        "timestamp_depth",
        "image",
        "depth_map"
    ]
)


def matrix_to_pose_parameters(G):
    R = G[0:3, 0:3]
    t = G[0:3, 3]
    q = rotation_to_quaternion(R)
    return t, q


def export_pose_sequence(filename, pose_sequence):
    # sort by timestamp
    pose_sequence = sorted(pose_sequence.items(), key=lambda x: x[0])
    with open(filename, "w") as f:
        writer = csv.writer(f, delimiter=' ')

        for timestamp, G in pose_sequence:
            # G is the absolute pose w.r.t the world coordinate
            t, q = matrix_to_pose_parameters(G)
            qw, qx, qy, qz = q

            writer.writerow([
                timestamp, t[0], t[1], t[2], qx, qy, qz, qw
            ])


class TUMDataset(object):
    def __init__(self, dataset_root, depth_factor=5000):
        self.dataset_root = dataset_root

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
