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

    with open(path, "r") as f:
        # read lines with ignoring comments
        reader = csv.reader(
            filter(lambda row: not is_comment(row), f),
            delimiter=' '
        )

        timestamps = []
        poses = []
        for row in reader:
            timestamps.append(row[0])
            poses.append([float(v) for v in row[1:]])
        poses = np.array(poses)

    return timestamps, poses


def list_images():
    path = Path(dataset_root, "rgb")
    return [p.name for p in path.glob("*.png")]


def load_rgb_image(filename):
    print(filename)
    path = Path(dataset_root, "rgb", filename)
    image = imread(str(path))
    print(image)


timestamps, poses = load_groundtruth()

load_rgb_image(timestamps[0])
filenames = list_images()
for filename in filenames:
    filename = filename.replace(".png", "")
    print(filename)
    print(timestamps)
    print(filename in timestamps)
