import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from skimage.color import rgb2gray
import numpy as np
from tqdm import tqdm

from tadataka import VisualOdometry, CameraParameters
from tadataka.rigid import exp_se3, log_se3
from tadataka.projection import warp
from tadataka.mapping import MapBuilder
from tadataka.quaternion import quaternion_to_rotation
from tadataka.datasets.tum_rgbd import TUMDataset, PoseSequence

from visualization.plot import plot


# dataset format is explained at
# https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats#
# intrinsic_camera_calibration_of_the_kinect

dataset_root = Path("datasets", "rgbd_dataset_freiburg1_desk")
# dataset_root = Path("datasets", "rgbd_dataset_freiburg2_pioneer_360")
# dataset_root = Path("datasets", "rgbd_dataset_freiburg3_structure_texture_near")


def error(image_true, image_pred, mask):
    return np.power(image_true[mask]-image_pred[mask], 2).mean()


def visualize_error_function(camera_parameters, I0, D0, I1, xi_pred):
    def generate_error_curve(i, start, stop, n):
        xi = np.copy(xi_pred)

        vs = xi[i] + np.linspace(start, stop, n)
        errors = []
        for v in vs:
            xi[i] = v
            DG = exp_se3(xi)
            estimated, mask = warp(camera_parameters, I1, D0, DG)
            errors.append(error(I0, estimated, mask))
        errors = np.array(errors)
        return vs, errors

    from matplotlib import pyplot as plt

    fig = plt.figure()

    for xi_index, ax_index in enumerate([1, 3, 5, 2, 4, 6]):
        ax = fig.add_subplot(3, 2, ax_index)

        vs, errors = generate_error_curve(xi_index,
                                          start=-0.10, stop=0.10, n=101)
        ax.set_title("Axis {}".format(xi_index+1))
        ax.axvline(vs[np.argmin(errors)], label="ground truth")
        ax.axvline(xi_pred[xi_index], color="red", label="prediction")
        ax.legend()
        ax.plot(vs, errors)

    plt.show()


def main():
    np.set_printoptions(suppress=True, precision=8, linewidth=1e8)

    camera_parameters = CameraParameters(
        focal_length=[525.0, 525.0],
        offset=[319.5, 239.5]
    )

    dataset = TUMDataset(dataset_root)

    G = np.eye(4)
    frame0 = dataset.load_color(0)

    sequence_pred = PoseSequence()
    sequence_pred.add(frame0.timestamp_depth, G)

    for i in tqdm(range(1, dataset.size)):
        frame1 = dataset.load_color(i)

        # TODO not necessary to convert the color of the same image twice
        # we need to create a better interface
        vo = VisualOdometry(camera_parameters,
                            rgb2gray(frame0.image), frame0.depth_map,
                            rgb2gray(frame1.image))
        DG = vo.estimate_motion(n_coarse_to_fine=6)

        G = np.dot(G, np.linalg.inv(DG))

        sequence_pred.add(frame1.timestamp_depth, G)

        frame0 = frame1

    sequence_pred.save("poses.txt")

    # TODO implement the following
    # pointcloud = map_builder.export()
    # export_pointcloud(pointcloud)

main()
