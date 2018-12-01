from pathlib import Path
import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from skimage.io import imread
import numpy as np

from motion_estimation import VisualOdometry, CameraParameters


dataset_root = Path("dataset", "rgbd_dataset_freiburg1_desk")

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
        self.timestamps = self.load_timestamps(self.rgb_root)

    def filename_to_timestamp(self, filename):
        return float(filename.replace(".png", ""))

    def timestamp_to_filename(self, timestamp):
        return str(timestamp) + ".png"

    def load_timestamps(self, directory):
        filenames = [p.name for p in directory.glob("*.png")]
        timestamps = [self.filename_to_timestamp(f) for f in filenames]
        return np.array(np.sort(timestamps))

    def search_nearest_timestamp(self, timestamp):
        index = np.searchsorted(self.timestamps, timestamp)
        return self.timestamps[index]

    def search_nearest(self, timestamp):
        nearest = self.search_nearest_timestamp(timestamp)
        filename = self.timestamp_to_filename(nearest)
        rgb_path = Path(self.rgb_root, filename)
        depth_path = Path(self.depth_root, filename)
        return str(rgb_path), str(depth_path)



def main():
    timestamps, poses = load_groundtruth()
    dataset = Dataset()

    query = timestamps[int(len(timestamps)/2)]
    print("Query: ", query)
    rgb_path, depth_path = dataset.search_nearest(query)
    print(rgb_path, depth_path)


main()
