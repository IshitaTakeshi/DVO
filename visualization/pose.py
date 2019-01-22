import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)


class PoseSequenseVisualizer(object):
    def __init__(self, sequences, axis_length=0.01):
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig)
        # self.ax = fig.gca(projection='3d')
        self.ax.set_aspect('equal')
        self.axis_colors = ["red", "green", "blue"]
        self.axis_length = axis_length
        self.visualize(sequences)

    def visualize(self, sequences):
        for i, sequence in enumerate(sequences):
            self.add_path(i, sequence)
            self.add_poses(sequence)
        set_axes_equal(self.ax)

    def add_path(self, i, sequence):
        ts = np.array([G[0:3, 3] for G in sequence])
        self.ax.plot(ts[:, 0], ts[:, 1], ts[:, 2],
                     label="sequence {}".format(i))

    def add_poses(self, sequence):
        for G in sequence:
            self.add_pose(G)

    def add_pose(self, G):
        R = G[0:3, 0:3]
        t = G[0:3, 3]

        for i in range(3):
            s = t
            d = t + self.axis_length * R[0:3, i]
            self.ax.plot([s[0], d[0]], [s[1], d[1]], [s[2], d[2]],
                         color=self.axis_colors[i])

    def show(self):
        plt.legend()
        plt.show()
