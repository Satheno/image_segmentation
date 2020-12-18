import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.distance import cdist
from scipy.spatial.distance import norm
import multiprocessing as mp
import random


class Visualizer:

    def __init__(self, n_pixels: int, inverted_grey_values: bool, alpha_voxels: bool):
        """
        Base class of Visualizer objects
        """
        self._n_pixels = n_pixels
        self._inverted_grey_values = inverted_grey_values
        self._alpha_voxels = alpha_voxels
        if self._inverted_grey_values:
            self._noise_color = "#33333350" if self._alpha_voxels else "#333333"
            self._point_color = "#00000050" if self._alpha_voxels else "#000000"
            self._empty_color = "#FFFFFF02" if self._alpha_voxels else "#FFFFFF"
        else:
            self._noise_color = "#CCCCCC50" if self._alpha_voxels else "#CCCCCC"
            self._point_color = "#FFFFFF50" if self._alpha_voxels else "#FFFFFF"
            self._empty_color = "#00000002" if self._alpha_voxels else "#000000"

    def draw_original(self, instance, azim, elev):
        pass

    def _color_points(self, instance, fileld, colors):
        pass

    def draw_pixel(self, instance):
        """
        Function used to draw the 3d voxel variant of the problem instance. Additionally the function calculates the
        3d matrix containing 8 bit values for the clustering process as well as the ground truth labels.

        :param instance: the problem instance (voronoi or curve instance)
        :return: 2-tuple containing the 8bit 3d matrix and the ground truth labels as flattened 3d matrix
        """
        dim = (self._n_pixels, self._n_pixels, self._n_pixels)
        colors = np.full(dim, self._empty_color)
        filled = np.zeros(dim, dtype=np.bool) if self._inverted_grey_values else np.ones(dim, dtype=np.bool)

        filled, colors, cluster_truth = self._color_points(instance, filled, colors)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        if self._alpha_voxels:
            # matplotlib bug workaround
            colors = self._explode(colors)
            filled = self._explode(filled)
            x, y, z = self._expand_coordinates(np.indices(np.array(filled.shape) + 1))

            ax.voxels(x, y, z, filled, facecolors=colors, edgecolors="#FFFFFF00", shade=False)
        else:
            ax.voxels(filled, facecolors=colors, edgecolors="#000000", shade=False)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()

        # convert 24 bit color matrix from matplotlib visualization to 8 bit integer value matrix
        eight_bit = np.zeros(colors.shape)
        for x_idx in range(colors.shape[0]):
            for y_idx in range(colors.shape[1]):
                eight_bit[x_idx][y_idx] = [int("0x" + elem[1:3], 0) for elem in colors[x_idx][y_idx]]

        return eight_bit.astype(int), cluster_truth

    def _explode(self, data):
        """
        Internal helper function to work around a bug in matplotlib 3d, where transparent voxels don't render as
        expected. Faces of voxels on the inside of structures are not rendered correctly.
        """
        shape_arr = np.array(data.shape)
        size = shape_arr[:3] * 2 - 1
        exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
        exploded[::2, ::2, ::2] = data
        return exploded

    def _expand_coordinates(self, indices):
        """
        Internal helper function to work around a bug in matplotlib 3d, where transparent voxels don't render as
        expected. Faces of voxels on the inside of structures are not rendered correctly.
        """
        x, y, z = indices
        x[1::2, :, :] += 1
        y[:, 1::2, :] += 1
        z[:, :, 1::2] += 1
        return x, y, z

    def _calculate_point_color(self, point_dist, max_dist, abs_max_rand=10, variant="voronoi"):
        rand_frac = random.randint(0, 2*abs_max_rand) - abs_max_rand
        if variant == "voronoi":
            color = 255 * (point_dist / max_dist) + 128 + rand_frac
        else:
            color = 255 * (point_dist / max_dist) - 128 + rand_frac
        color = 255 if color > 255 else color
        color = 0 if color < 0 else color
        if self._inverted_grey_values:
            if variant == "voronoi":
                color = int(255 - color)
            else:
                color = int(color)
        else:
            if variant == "voronoi":
                color = int(color)
            else:
                color = int(255 - color)
        color = hex(color)[2:]
        if len(color) == 1:
            color = "0" + color
        return color


class CurveVisualizer(Visualizer):

    def __init__(self, n_pixels: int, noise=6, inverted_grey_values: bool = True, alpha_voxels: bool = True):
        """
        Curve Visualizer object used to draw problem instances generated by the CurveDatasetGenerator. Two different
        ways of visualization are provided, e.g. the standard matplotlib visualization of curves using just the points
        of every curve in a problem instance (see draw_original) and a voxel based approximation of the curves with a
        resolution of <n_pixels> on every axis, i.e. <n_pixels>^3 voxels in total (see draw_pixel). The visualization
        scales up the problem instance to the given <n_pixels> on every axis.

        :param n_pixels: The number of pixels/voxels per axis in the draw_pixel visualization
            CAUTION: The rendering process will take some time, especially when using alpha_voxels, because the function
            used a bug workaround in this case. Remember that the arrays grow cubic and through the workaround quintic i
            guess. Keep that in mind and use small values. 16, 32, 48 or 64 is fine but above could be problematic.
        :param noise: Whether the draw_pixel visualization should have some noise around the approximation or not.
            Values can be [None, 6, 27, 32]
        :param inverted_grey_values: Whether the grey values for the visualization should be inverted or not. The
            default (True) only draws the curves with dark voxels and non-inverted values (False) will draw the curves
            white and their surrounding pixels black. (In the latter you wont see the curves without using alpha_voxels)
        :param alpha_voxels: Whether to use alpha on the voxels or not. If used (True) this will cause the function to
            use a bug workaround which increases the execution time due to higher dimensionality of matrices (4 instead
            of 3)
        """
        super().__init__(n_pixels, inverted_grey_values, alpha_voxels)
        assert noise in [None, 6, 26, 32]
        self._noise = noise
        # TODO: Implement choice whether to use voxels with alpha or not (variant without alpha still missing)

    def draw_original(self, instance, azim=None, elev=None):
        """
        Simple function which draws the curves given in the instance the way they are defined, i.e. as lines.

        :param instance: The instance (dict) containing the curves to draw. keys should represent the curve ids and the
            associated values should be of shape (number_of_points_per_curve, 3)
        :param azim: Parameter for the view_init function of matplotlib
        :param elev: Parameter for the view_init function of matplotlib
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for curve_id, curve_points in instance.items():
            plt.plot(curve_points[:, 0], curve_points[:, 1], curve_points[:, 2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(azim=azim, elev=elev)
        plt.show()

    def _color_point(self, colors, filled, clustering_truth, cluster_idx, max_dist, point, offset=np.array([0, 0, 0])):
        """
        Internal function used to color voxels in the draw_pixel function and to color the noise voxels

        :param colors: The np.array containing the HTML color strings for every voxel
        :param filled: The np.array containing the presence information for every voxel (if a voxel is drawn or not)
        :param clustering_truth: The np.array containing the ground truth labels for the points
        :param cluster_idx: The index of the current cluster
        :param point: The coordinates of a voxel for the use in the colors/filled np.array
        :param color: The color the point will have in HTML format with alpha value --> #RRGGBBAA or #RRGGBB
        :param offset: The offset for the point. This is currently only used for adding noise
        :return: the altered colors, filled np.array
        """
        offset_point = point + offset
        if max(offset_point) < (len(colors) - 1) and min(offset_point) > 0 and \
                colors[offset_point[0]][offset_point[1]][offset_point[2]] != self._point_color:
            color = self._calculate_point_color(norm(point - offset_point), max_dist, variant="curve")
            colors[offset_point[0]][offset_point[1]][offset_point[2]] = \
                "#" + 3 * color + "50" if self._alpha_voxels else "#" + 3 * color
            filled[offset_point[0]][offset_point[1]][offset_point[2]] = 1
            clustering_truth[offset_point[0], offset_point[1], offset_point[2]] = cluster_idx
        return colors, filled, clustering_truth

    def _color_points(self, instance, filled, colors):
        """
        Internal function used by the draw_pixel function to calculate the voxel positions that will be drawn, the
        colors of these voxels and the ground truth labels for the curve problem instance.

        :param instance: curve problem instance
        :param filled: np.array of shape (x,x,x) containing booleans indicating if the voxel at this position should
            be drawn or not
        :param colors: np.array of shape (x,x,x) containing the color strings for each voxel
        :return: 3-tuple containing the filled array, colors array and the ground truth labels array
        """
        clustering_truth = np.full(filled.shape, 99999)
        for cluster_idx, (_, curve_points) in enumerate(instance.items()):
            curve_points *= self._n_pixels
            curve_points = curve_points.astype(np.int)
            for point in curve_points:
                colors, filled, clustering_truth = self._color_point(colors, filled, clustering_truth, cluster_idx, 1,
                                                                     point)
                # add noise (the 6 cubes around a point in this case)
                if self._noise == 6:
                    neighbors = np.array([[0, 0, -1], [0, 0, +1], [0, -1, 0],
                                          [0, +1, 0], [-1, 0, 0], [+1, 0, 0]])
                    max_dist = norm([1, 0, 0])
                elif self._noise == 27:
                    neighbors = np.array([[-1, -1, -1], [-1, -1, ], [-1, -1, +1], [-1, 0, -1], [-1, 0, 0], [-1, 0, +1],
                                          [-1, +1, -1], [-1, +1, 0], [-1, +1, +1], [0, -1, -1], [0, -1, 0], [0, -1, +1],
                                          [0, 0, -1], [0, 0, +1], [0, +1, -1], [0, +1, 0], [0, +1, +1], [+1, -1, -1],
                                          [+1, -1, 0], [+1, -1, +1], [+1, 0, -1], [+1, 0, 0], [+1, 0, +1], [+1, +1, -1],
                                          [+1, +1, 0], [+1, +1, +1]])
                    max_dist = norm([1,1,1])
                elif self._noise == 32:
                    neighbors = np.array([[-1, -1, -1], [-1, -1, ], [-1, -1, +1], [-1, 0, -1], [-1, 0, 0], [-1, 0, +1],
                                          [-1, +1, -1], [-1, +1, 0], [-1, +1, +1], [0, -1, -1], [0, -1, 0], [0, -1, +1],
                                          [0, 0, -1], [0, 0, +1], [0, +1, -1], [0, +1, 0], [0, +1, +1], [+1, -1, -1],
                                          [+1, -1, 0], [+1, -1, +1], [+1, 0, -1], [+1, 0, 0], [+1, 0, +1], [+1, +1, -1],
                                          [+1, +1, 0], [+1, +1, +1], [0, 0, -2], [0, 0, +2], [0, -2, 0], [0, +2, 0],
                                          [-2, 0, 0], [+2, 0, 0]])
                    max_dist = norm([1, 1, 1])
                if self._noise is not None:
                    for offset in neighbors:
                        colors, filled, clustering_truth = self._color_point(colors, filled, clustering_truth,
                                                                             cluster_idx, max_dist, point, offset=offset)
        return filled, colors, clustering_truth.flatten()


class VoronoiVisualizer(Visualizer):

    def __init__(self, n_pixels, inverted_grey_values: bool = True, max_dist: float = .005, alpha_voxels: bool = True):
        """
        Voronoi Visualizer object used to draw problem instances generated by the VoronoiDatasetGenerator. Two
        different ways of visualization are provided, e.g. the standard matplotlib visualization where you can
        choose a combination of boundaries, center nodes and faces of the voronoi cells to visualize and a voxel
        based approximation visualization of the voronoi cells (through the boundaries) with a resolution of
        <n_pixels> on every axis, i.e. <n_pixels>^3 voxels in total (see draw_pixel). The visualization scales up
        the problem instance to the given <n_pixels> on every axis.

        :param n_pixels: The number of pixels/voxels per axis in the draw_pixel visualization
            CAUTION: The np.arrays used to render the visualization will get huge, because of a bug workaround i used.
            Remember that the arrays grow cubic and through the workaround quintic i guess. Keep that in mind and use
            small values. 16, 32, 48 or 64 is fine but above could be problematic.
        :param inverted_grey_values: Whether the grey values for the visualization should be inverted or not. The
            default (True) only draws the voronoi cell boundaries black in the interior of the cells white. Non-inverted
            grey values will draw the interior of the cells black and the boundaries white
        :param alpha_voxels: Whether to use alpha on the voxels or not. If used (True) this will cause the function to
            use a bug workaround which increases the execution time due to higher dimensionality of matrices (4instead
            of 3)
        """
        super().__init__(n_pixels, inverted_grey_values, alpha_voxels)
        self._max_dist = max_dist

    def draw_original(self, instance, azim=None, elev=None, boundary=False, center_nodes=False, faces=True):
        """
        Function used to draw a voronoi problem instance. This is the standard matplotlib visualization. You can choose
        if you want to draw the boundaries, center nodes and faces.
        Currently 22 different cells can be drawn. If you want more, extend the colors list at the beginning of this
        function.

        :param instance: The voronoi problem instance
        :param azim: Parameter for the view_init function of matplotlib
        :param elev: Parameter for the view_init function of matplotlib
        :param boundary: Whether to draw boundaries or not
        :param center_nodes: Whether to draw center nodes of the voronoi cells (subcell centers will be drawn also)
        :param faces: Whether to draw cell faces or not
        """
        colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c',
                  '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
                  '#000075', '#808080', '#ffffff', '#000000']
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for c_idx, cell in enumerate(instance):

            if boundary:
                # draw boundaries
                for v_start_idx, adj_idxs in enumerate(cell["adjacency"]):
                    for v_end_idx in adj_idxs:
                        line = np.append([cell["vertices"][v_start_idx]], [cell["vertices"][v_end_idx]], axis=0)
                        ax.plot(line[:, 0], line[:, 1], line[:, 2], c="#000000", ls="-", lw=1.)

            if center_nodes:
                # draw center node
                assert len(instance) <= len(colors)
                if len(cell["original"].shape) == 1:
                    center = np.array([cell["original"]])
                else:
                    center = cell["original"]
                ax.scatter(center[:, 0], center[:, 1], center[:, 2], c=colors[c_idx])

            if faces:
                # draw faces
                assert len(instance) <= len(colors)
                for face_dict in cell["faces"]:
                    points = np.array([[cell["vertices"][v_idx] for v_idx in face_dict["vertices"]]])
                    ax.add_collection3d(Poly3DCollection(points, alpha=.2, fc=colors[c_idx]))

        # axis labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        # set view
        ax.view_init(azim=azim, elev=elev)
        plt.show()

    def _color_points_parallel(self, instance, filled, colors, base_voxel, voxel_offset, originals, originals_idxs,
                               exec_idx):
        """
        Internal function used to calculate the colors, filled state and ground truth labels for the voronoi instance in
        parallel. Each function calculates 1/cores of the total amount of layers in the visualization.

        :param instance: the voronoi problem instance
        :param filled: np.array containing booleans indicating whether the voxel at each position should be drawn or not
        :param colors: np.array containing string representing the colors of each voxel
        :param base_voxel: np.array containing voxel coordinates that will be used to calculate the current voxel
            position
        :param voxel_offset: np.array containing the offset to obtain the center of the voxel for distance calculation
        :param originals: np.array of shape (num_centers, 3) containing all centers of the voronoi cells (used to
            identify the corresponding voronoi cell and subcell for each voxel, thus making distance calculation to the
            nearest subcell face possible)
        :param originals_idxs: np.array containing indices num_subcells entries of the cell_idx for every cell in
            ascending cell order (used to identify the subcell center in a cell)
        :param exec_idx: integer used to order the output of this function as this function is executed asynchronous
        :return: 4-tuple containing filled array, colors array, ground truth array, exec_idx
        """
        cluster_truth = np.full(filled.shape, 99999)
        dim = filled.shape[0]
        xy_dim = filled.shape[1]
        for z_idx in range(dim):
            for y_idx in range(xy_dim):
                for x_idx in range(xy_dim):
                    current_voxel = base_voxel * np.array([x_idx, y_idx, z_idx]) + voxel_offset

                    # find cell and then subcell of voxel [x_idx, y_idx, z_idx]
                    original_distances = cdist(np.array([current_voxel]), originals)[0]
                    originals_min_idx = np.argmin(original_distances)
                    cell_idx = int(originals_idxs[originals_min_idx])
                    subcell_idx = int(originals_min_idx - np.where(originals_idxs == cell_idx)[0][0])

                    # add label to current voxel (x_idx, y_idx, z_idx) for ground truth matrix
                    cluster_truth[z_idx][y_idx][x_idx] = cell_idx

                    # find all faces belonging to the subcell and calculate the planes of these
                    if len(instance[cell_idx]["original"].shape) > 1:
                        # at least two original points in "original" array --> use subcell_idx
                        subcell_faces = [face["vertices"] for face in instance[cell_idx]["faces"] if
                                         face["adjacent_cell"] >= 0 and
                                         np.allclose(face["original"], instance[cell_idx]["original"][subcell_idx])]
                    else:
                        # only one original point in "original" array --> don't use subcell_idx
                        subcell_faces = [face["vertices"] for face in instance[cell_idx]["faces"] if
                                         face["adjacent_cell"] >= 0 and
                                         np.allclose(face["original"], instance[cell_idx]["original"])]
                    if len(subcell_faces) == 0:
                        # if there are no subcell faces then this cell is far from all boundaries -> voxel is white
                        continue
                    subcell_face_planes = []
                    cell = instance[cell_idx]
                    for v in subcell_faces:
                        cp = np.cross(cell["vertices"][v[2]] - cell["vertices"][v[0]],
                                      cell["vertices"][v[1]] - cell["vertices"][v[0]])
                        cp = cp / norm(cp)
                        subcell_face_planes.append(np.array([cp[0], cp[1], cp[2], cp.dot(cell["vertices"][v[2]])]))

                    # calculate distances to faces and color based on this distance
                    voxel_plane_dist = np.apply_along_axis(
                        lambda p: abs((p[:3].dot(current_voxel) - p[3]) / ((sum(p[:3] ** 2)) ** .5)),
                        1, subcell_face_planes)
                    min_dist = min(voxel_plane_dist)
                    if min_dist < self._max_dist:
                        cluster_truth[z_idx][y_idx][x_idx] = 99999  # because its a wall and therefore not in a cluster
                        color = self._calculate_point_color(min_dist, self._max_dist)
                        filled[z_idx][y_idx][x_idx] = True
                        colors[z_idx][y_idx][x_idx] = "#" + 3 * color + "50" if self._alpha_voxels else "#" + 3 * color
        return filled, colors, cluster_truth, exec_idx

    def _color_points(self, instance, filled: np.array, colors: np.array):
        """
        Internal function used by the draw_pixel function to calculate the filled array, colors array and ground truth
        labels. As this function takes quite long the execution is parallelized but still quite slow.

        :param instance: The voronoi problem instance
        :param filled: np.array containing booleans indicating whether the voxel at each position should be drawn or not
        :param colors: np.array containing string representing the colors of each voxel
        :return: 3-tuple containing filled array, colors array and flattened ground truth label array
        """
        dim = filled.shape[0]
        originals = np.array([])
        originals_idxs = np.array([])
        for idx, cell in enumerate(instance):
            tmp_rows = [cell["original"]] if len(cell["original"].shape) == 1 else cell["original"]
            if len(originals) == 0:
                originals = tmp_rows
            else:
                originals = np.append(originals, tmp_rows, axis=0)
            originals_idxs = np.append(originals_idxs, np.full((len(tmp_rows),), idx))

        base_voxel = np.full((3,), 1 / dim)
        voxel_offset = np.full((3,), .5 / dim)

        # Hyperthreading is enabled --> use just physical cores, i.e. divide cpu_count by 2
        cpu_count = int(mp.cpu_count() / 2)

        layer_per_core = int(dim / cpu_count)
        layer_idxs = np.arange(0, dim, layer_per_core)
        layer_idxs[-1] = dim
        pool = mp.Pool(processes=cpu_count)
        results = [pool.apply_async(self._color_points_parallel, args=(
            instance, filled[layer_idxs[i]:layer_idxs[i + 1]], colors[layer_idxs[i]:layer_idxs[i + 1]], base_voxel,
            voxel_offset + i * layer_per_core * np.array([0, 0, 1 / dim]), originals, originals_idxs, i)) for i in
                   range(len(layer_idxs) - 1)]
        outputs = sorted([result.get() for result in results], key=lambda elem: elem[3])
        filled = None
        colors = None
        clustering_truth = None
        # build return values from outputs
        for out in outputs:
            if filled is None:
                filled = np.swapaxes(out[0], 0, 2)
            else:
                filled = np.append(filled, np.swapaxes(out[0], 0, 2), axis=2)
            if colors is None:
                colors = np.swapaxes(out[1], 0, 2)
            else:
                colors = np.append(colors, np.swapaxes(out[1], 0, 2), axis=2)
            if clustering_truth is None:
                clustering_truth = np.swapaxes(out[2], 0, 2)
            else:
                clustering_truth = np.append(clustering_truth, np.swapaxes(out[2], 0, 2), axis=2)

        return filled, colors, clustering_truth.flatten()
