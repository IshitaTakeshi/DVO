import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tadataka.rigid import transform
from tadataka.projection import inverse_projection


class MapBuilder(object):
    def __init__(self, camera_parameters):
        # represent the map as a set of points
        self.points = []
        self.colors = []
        self.camera_parameters = camera_parameters

    def update(self, G, color_image, depth_map, mask=None):
        isvalid = depth_map.flatten() > 0

        points = inverse_projection(self.camera_parameters, depth_map)
        points = transform(G, points[isvalid])
        self.points.append(points)

        colors = color_image.reshape(-1, 3)
        self.colors.append(colors[isvalid])

    def show(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        points = np.vstack(self.points)
        colors = np.vstack(self.colors)
        colors = colors.astype(np.float64) / 255.
        ax.scatter(points[:, 0], points[:, 1], points[:, 1], c=colors)
        plt.show()
