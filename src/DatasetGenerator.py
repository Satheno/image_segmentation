import numpy as np
import pyvoro as pv
import random
import pickle
import os
import splipy.curve_factory as curve_factory
from scipy.spatial.distance import cdist
from itertools import combinations
from tqdm import tqdm


class DatasetGenerator:

    def __init__(self, num_instances: int, seed: float = 42):
        """
        Base class for DatasetGenerators
        """
        self._num_instances = num_instances
        self._seed = seed
        # apply seed
        random.seed(self._seed)
        np.random.seed(self._seed)

    def generate_dataset(self):
        dataset = []
        for i in range(self._num_instances):
            dataset.append(self._generate_instance())
        return dataset

    def _generate_instance(self):
        pass

    def generate_and_save_dataset(self, folder: str):
        # just in case the user forgot to add a / to the end of his path
        if folder[-1] != "/":
            folder += "/"

        dataset, description = self.generate_dataset()

        # save every problem instance
        for instance_idx in range(len(dataset)):
            with open(folder + f"instance_{instance_idx}.pkl", "wb") as file:
                pickle.dump(dataset[instance_idx], file, protocol=pickle.HIGHEST_PROTOCOL)

        # save the dataset description
        with open(folder + "description.pkl", "wb") as file:
            pickle.dump(description, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_dataset(self, folder: str):
        pass


class CurveDatasetGenerator(DatasetGenerator):

    def __init__(self, num_instances: int, num_curves: int, num_eval_points: int, num_control_points: int,
                 min_distance: float, min_length: int, seed: float = 42):
        """
        CurveDataSetGenerator object which is used to generate a dataset containing <num_instances> problem instances
        which contain <num_cuves> different generated curves based on <num_control_points> and which are sampled along
        <num_eval_points> uniformly spaced points. Each pair of generated curves has at least a minimal distance of
        <min_distance> and at least <min_length> points located inside the unit cube. The whole generation process is
        seed-based and can reproduce datasets when the same parameters are given. See generate_dataset for usage.

        :param num_instances: The number of different problem instances that will be generated
        :param num_curves: The number of different curves in each problem instance
        :param num_eval_points: The number of uniformly spaced points which will be used to approximate the curve
        :param num_control_points: The number of control points used to specify the shape of the curve
        :param min_distance: The minimal distance between every pair of curves
        :param min_length: The minimal amount of points of the eval_points that must be located inside the unit cube
        :param seed: float seed used for the generation process
        """
        super().__init__(num_instances, seed)
        self._num_curves = num_curves
        self._num_eval_points = num_eval_points
        self._num_control_points = num_control_points
        self._min_distance = min_distance
        self._min_length = min_length

    def generate_dataset(self):
        """
        Generates dataset according to the given parameters with which this object was created.

        :return: Tuple with 2 elements, the first one is the dataset (list of instances, each one is a dict containing
            the curve_ids as keys and curve_points as associated values in a list) and the second is a description of
            the dataset (a dict containing the parameter names as keys and associated parameter values as dict values)
        """
        dataset = []
        for i in range(self._num_instances):
            dataset.append(self._generate_instance())
        description = {"num_instances": self._num_instances,
                       "num_curves": self._num_curves,
                       "num_eval_points": self._num_eval_points,
                       "num_control_points": self._num_control_points,
                       "min_distance": self._min_distance,
                       "min_length": self._min_length,
                       "seed": self._seed}
        return dataset, description

    def _generate_instance(self):
        """
        Internal function of the CurveDatasetGenerator that generates a problem instance containing <_num_curves>
        generated curves based on <_num_control_points> Control points along <_num_eval_points> uniformly spaced points
        with at least a minimal distance of <_min_distance> between each pair of curves.
        
        :return: Instance of curves as dict, where keys are the string names of the curves and the associated values are
            lists of points representing the curves
        """
        instance = {}

        for i in range(self._num_curves):
            curve_points = np.array([])
            valid_curve = False
            # The while loop will be executed until a valid curve is found. Each iteration of the for loop will produce
            # one valid curve, but it may take a while with large numbers in self._num_curves due to multiple 
            # iterations of the while loop 
            while not valid_curve:
                controlpoints = np.array(
                    [np.array([random.uniform(-.5, 1.5), random.uniform(-.5, 1.5), random.uniform(-.5, 1.5)]) for _ in
                     range(self._num_control_points)])
                curve = curve_factory.cubic_curve(controlpoints, 4)
                eval_points = np.linspace(0, 1, self._num_eval_points)
                curve_points = np.array(curve(eval_points))

                # get all points outside of the unit cube and delete them
                rows_to_delete = np.append(np.where(curve_points < 0)[0], np.where(curve_points > 1)[0])
                reference_rows = np.array(range(len(curve_points)))
                reference_rows = np.delete(reference_rows, rows_to_delete)
                # checks whether the generated curve is valid or not
                if len(reference_rows) < self._min_length or max(np.diff(reference_rows)) > 1:
                    # All points are out of the unit cube or the segment which is outside of the unit cube which will be
                    # deleted splits the curve in half, thus creating two seperate curve segments in the unit cube
                    continue

                curve_points = np.delete(curve_points, rows_to_delete, axis=0)

                if len(instance) == 0:
                    # First generated curve, therefore it cant be invalid in terms of minimal distance to other curves
                    break

                for tmp_curve_id, tmp_curve_points in instance.items():
                    distances = cdist(curve_points, tmp_curve_points)
                    if np.min(distances) <= self._min_distance:
                        # Generates new curve, because the current one is too close to at least one other curve
                        valid_curve = False
                        break
                    # valid_curve is only true if the distances to all other curves is at least <_min_distance>
                    valid_curve = True

            instance[f"curve_{i}"] = curve_points
        return instance

    def load_dataset(self, folder: str):
        """
        Function that loads a curve dataset from a folder that was previously saved with the generate_and_save_dataset
        function. Additionally the function checks some parameters with which the dataset was created and raises an
        Error if at least one parameter doesn't match

        :param folder: The folder containing the voronoi dataset in different instance (.pkl) files and a description
            (.pkl) file
        :return: 2-tuple containing the dataset (list of instances) and the description (dictionary)
        """
        # just in case the user forgot to add a / to the end of his path
        if folder[-1] != "/":
            folder += "/"

        files = os.listdir(folder)
        description_file = [name for name in files if name.__contains__("description")]
        instance_files = sorted([name for name in files if name.__contains__("instance")])

        # make sure only one description file exists
        assert len(description_file) == 1

        description = None
        description_file = description_file[0]
        with open(folder + description_file, "rb") as file:
            description = pickle.load(file)

        dataset = []
        for instance_file in instance_files:
            with open(folder + instance_file, "rb") as file:
                dataset.append(pickle.load(file))

        # test if everything is alright (tests the easy parameters for every instance)
        assert len(dataset) == description["num_instances"]
        for instance in dataset:
            assert len(instance) == description["num_curves"]
            # check parameters for curves
            for curve_id, curve_points in instance.items():
                assert len(curve_points) <= description["num_eval_points"]
                assert len(curve_points) >= description["min_length"]

        return dataset, description


class VoronoiDatasetGenerator(DatasetGenerator):

    def __init__(self, num_instances: int, num_start_cells: int, num_end_cells: int, block_size: float = .01,
                 seed: float = 42):
        '''
        VoronoiDatasetGenerator object used to generate <num_instances> problem instances each starting with
        <num_starting_cells> voronoi cells and then merging them until there are <num_end_cells> voronoi cells left.
        The cells will be generated in the unit cube with a block size of <block_size> in every dimension. Keep in mind
        that the generation process may take a while, especially with a big number of <num_starting_cells>.

        :param num_instances: int number of instances that should be generated in the dataset
        :param num_start_cells: int number of voronoi cells to start with
        :param num_end_cells: int number of voronoi cells that should be left after the merging process
        :param block_size: float number representing the block size for the generation of the voronoi tesselation
        :param seed: float seed for the generation process
        '''
        super().__init__(num_instances, seed)
        assert num_start_cells >= num_end_cells
        self._num_start_cells = num_start_cells
        self._num_end_cells = num_end_cells
        self._block_size = block_size

    def generate_dataset(self):
        '''
        Function used to generate a dataset containing problem instances according to the given parameters when this
        object was created

        :return: 2-tuple containing the dataset and the description. The dataset is a list of instances, each containing
            a voronoi tesselation based on the format from pyvoro. The description is a dict containing all parameters
            which were used to generate the dataset.
        '''
        dataset = []
        for i in range(self._num_instances):
            dataset.append(self._generate_instance())
        description = {"num_instances": self._num_instances,
                       "num_start_cells": self._num_start_cells,
                       "num_end_cells": self._num_end_cells,
                       "block_size": self._block_size}
        return dataset, description

    def _generate_instance(self, verbose: bool = False):
        '''
        Internal function used by the generate_dataset and generate_and_save_dataset function to generate problem
        instances according to the given parameters when the VoronoiDataSetGenerator was created.

        :param verbose: bool which determines whether to print information about the generation process, i.e. the merge
            count or cells which are merged at the moment

        :return: A problem instance as a list in the form of the return value from pyvoro
        '''
        start_points = np.array(
            [[random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)] for _ in range(self._num_start_cells)])
        instance = pv.compute_voronoi(points=start_points, limits=np.array([[0, 1] for _ in range(3)]),
                                      dispersion=self._block_size)

        # Append the center point of every cell to all its faces (necessary to identify closest face to pixel in vis)
        for cell in instance:
            for face in cell["faces"]:
                face["original"] = cell["original"]

        # merge cells until only <_num_end_cells> cells are left
        merge_count = 0
        for _ in tqdm(range(len(instance) - self._num_end_cells), "Instance Cell Merging"):
            merge_count += 1
            # choose cell begin with
            cell_idx_01 = random.randint(0, len(instance) - 1)
            # choose cell to merge with from other adjacent cells to cell at cell_idx_01
            cell_idx_02 = random.choice(
                [face["adjacent_cell"] for face in instance[cell_idx_01]["faces"] if
                 face["adjacent_cell"] >= 0 and face["adjacent_cell"] != cell_idx_01])
            if verbose:
                print(f"{merge_count}. Merge with cells {cell_idx_01} and {cell_idx_02}")
            instance = self._merge_cells(instance, cell_idx_01, cell_idx_02)
            instance = self._cleanup_instance(instance)
        return instance

    def _cleanup_remove_list(self, instance, cell_idx):
        '''
        Internal function used by the _cleanup_instance function. The function calculates a list of potential vertices
        to remove from the problem instance. A vertex can be removed if it is not a vertex on the boundary (see
        corner_points) or if it does not occur in a face to another cell (which is not the cell itself).

        :param instance: The problem instance as a list of voronoi cells
        :param cell_idx: The index of the current voronoi cell in the instance that should be examined
        :return: List of potential vertex indices to remove
        '''
        corner_points = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0],
                                  [1, 1, 0], [1, 1, 1], [0, 1, 1], [1, 0, 1]])
        # find and remove points which are in a cell but not in a face of this cell
        remove_list = []
        # find points to remove
        for v_idx in range(len(instance[cell_idx]["vertices"])):
            if not np.any([np.allclose(corner_points[i], instance[cell_idx]["vertices"][v_idx]) for i in range(8)]):
                # v_idx is not the index of a corner point (we want to keep corner points as boundary)
                remove_list.append(v_idx)
            else:
                # v_idx is the index of a corner point
                continue
            for face in instance[cell_idx]["faces"]:
                if face["adjacent_cell"] >= 0 and face["adjacent_cell"] != cell_idx and v_idx in face["vertices"] or \
                        len(instance[cell_idx]["adjacency"][v_idx]) == 0:
                    # v_idx is the index of a point in a face to another cell (not boundary) --> keep it
                    remove_list.remove(v_idx)
                    break
        return remove_list

    def _cleanup_adjacency(self, instance: list, cell_idx: int, remove_idx: int):
        '''
        Internal function used by the _cleanup_instance function. This function decrements every vertex index that is
        higher than the index of the vertex that will be removed (-> remove_idx) in the cell at the given index.
        This is necessary because by deleting the vertex at remove_idx every vertex with a higher index will move up one
        index.

        :param instance: The problem instance as a list containing the voronoi cells
        :param cell_idx: The integer index of the voronoi cell in which a vertex should be deleted
        :param remove_idx: The integer index of the vertex that will be deleted
        '''
        # update adjacency of every vertex
        for adj_idx in range(len(instance[cell_idx]["adjacency"])):
            new_adjacency = []
            for v_idx in instance[cell_idx]["adjacency"][adj_idx]:
                if v_idx > remove_idx:
                    new_adjacency.append(v_idx - 1)
                elif v_idx < remove_idx:
                    new_adjacency.append(v_idx)
            instance[cell_idx]["adjacency"][adj_idx] = new_adjacency

        # delete adjacency of remove_idx
        del instance[cell_idx]["adjacency"][remove_idx]

    def _cleanup_faces(self, instance, cell_idx, remove_idx):
        '''
        Internal function used by the _cleanup_instance function. This function decrements every vertex index higher
        than the index of the vertex that was removed. This is necessary because every vertex with a higher index than
        the vertex at remove_idx moved up one index making the current indices in the faces invalid.

        :param instance: The problem instance as a list containing the voronoi cells
        :param cell_idx: The integer index of the voronoi cell
        :param remove_idx: The integer index of the vertex that was deleted before
        '''
        # update vertices in faces
        for face_idx in range(len(instance[cell_idx]["faces"])):
            new_vertices = []
            for v_idx in instance[cell_idx]["faces"][face_idx]["vertices"]:
                if v_idx > remove_idx:
                    new_vertices.append(v_idx - 1)
                elif v_idx < remove_idx:
                    new_vertices.append(v_idx)
            instance[cell_idx]["faces"][face_idx]["vertices"] = new_vertices

    def _cleanup_instance(self, instance):
        '''
        Internal function used by the generate_dataset function to cleanup the instance after merging cells, i.e.
        removing redundant vertices or removing unwanted adjacency lists. This speeds up the visualization process and
        makes the visualization and tesselation itself cleaner. As a first step a list of potential vertices to remove
        is calculated. After that the vertices in the remove_list are removed or kept based on where they lie. Vertices
        on the boundary are always kept, because deleting them would require a recalculation of the faces they were in.
        This problem is NP-complete for non-convex polygons which is the case here. Thats why we keep these vertices and
        just remove all adjacent vertices from their adjacency lists as well as all occurrences of these vertices from
        other adjacency lists. Vertices in cells that don't occur in any face or faces to the cell itself will be
        deleted and the associated adjacency list with the vertex. All other occurrences of the deleted vertex will also
        be removed from other adjacency lists.

        :param instance: The problem instance with two previously joined cells as a list of voronoi cells
        :return: The cleaned up problem instance as a list of voronoi cells
        '''
        for cell_idx in range(len(instance)):
            remove_list = sorted(self._cleanup_remove_list(instance, cell_idx), reverse=True)

            for v_idx in remove_list:

                # replace/delete faces based on what/where they are
                v_idx_faces = [face for face in instance[cell_idx]["faces"] if v_idx in face["vertices"]]
                num_cells = np.unique([face["adjacent_cell"] for face in v_idx_faces])
                is_boundary = False
                for adj_cell in num_cells:
                    # check if current adj_cell is boundary
                    if adj_cell < 0:
                        is_boundary = True

                        # find the adjacent vertices of remove_idx which are both on the boundary and add new adjacency
                        for v_idx_01, v_idx_02 in combinations(instance[cell_idx]["adjacency"][v_idx], 2):
                            if sum(np.isclose(instance[cell_idx]["vertices"][v_idx_01],
                                              instance[cell_idx]["vertices"][v_idx_02])) == 2:
                                # found the pair on the boundary grid
                                instance[cell_idx]["adjacency"][v_idx_01].append(v_idx_02)
                                instance[cell_idx]["adjacency"][v_idx_02].append(v_idx_01)

                        # empty the adjacency list of v_idx
                        instance[cell_idx]["adjacency"][v_idx] = []

                        # remove all occurrences of v_idx in all adjacency_lists
                        for adj_idx in range(len(instance[cell_idx]["adjacency"])):
                            instance[cell_idx]["adjacency"][adj_idx] = [idx for idx in
                                                                        instance[cell_idx]["adjacency"][adj_idx] if
                                                                        idx != v_idx]

                    elif adj_cell == cell_idx:
                        # face to self; inside cell face --> delete it
                        inside_cell_faces_idxs = sorted([idx for idx, face in enumerate(instance[cell_idx]["faces"]) if
                                                         face["adjacent_cell"] == cell_idx], reverse=True)
                        for inside_face_idx in inside_cell_faces_idxs:
                            del instance[cell_idx]["faces"][inside_face_idx]

                        if not is_boundary:
                            # update adjacency lists f or all vertices with index > v_idx
                            self._cleanup_adjacency(instance, cell_idx, v_idx)

                            # update all vertices in faces for vertices with index > v_idx
                            self._cleanup_faces(instance, cell_idx, v_idx)

                            # delete vertex at v_idx
                            del instance[cell_idx]["vertices"][v_idx]

                    else:
                        print("Unknown Face, not negative and not cell_idx")

                # remove vertex if it does not occur in any face (happens from time to time)
                remove_vertex = True
                for face in instance[cell_idx]["faces"]:
                    if v_idx in face["vertices"]:
                        remove_vertex = False
                if remove_vertex:
                    # update adjacency lists f or all vertices with index > v_idx
                    self._cleanup_adjacency(instance, cell_idx, v_idx)

                    # update all vertices in faces for vertices with index > v_idx
                    self._cleanup_faces(instance, cell_idx, v_idx)

                    # delete vertex at v_idx
                    del instance[cell_idx]["vertices"][v_idx]

        return instance

    def _merge_c2_c1_mapping(self, instance: list, cell_idx_01: int, cell_idx_02: int, face_vertices: list):
        '''
        Internal function used by the _merge_cells function to generate the mapping from vertices indices of cell 02
        to vertices indices of cell 01 after they will be merged. This is necessary because we can't just append all
        vertices from cell 02 to cell 01, because both cells contain cells which are the same but with different indices
        in the vertices list. If a vertex (from cell 02) already exists in cell 01 then the index of this vertex (in the
        vertices list for cell 01) will be used for the same vertex in cell 02. If a vertex (from cell 02) doesn't exist
        in cell 01 then this vertex will be appended to the list and the index can be calculated using the length of the
        current vertices list of cell 01 and an offset (-> idx_offset).

        :param instance: The problem instance as a list containing all voronoi cells
        :param cell_idx_01: The integer index of the first cell (this cell will be kept later)
        :param cell_idx_02: The integer index of the second cell (this cell will be deleted later)
        :param face_vertices: A list of vertices which contains the vertices of the face where the two cells merged,
            i.e. a list of vertices that exist in both cells
        :return: dict containing the mapping from cell 02 indices (keys) to cell 01 indices (associated values)
        '''
        c2_in_c1_mapping = {}
        idx_offset = 0
        for v_idx_02 in range(len(instance[cell_idx_02]["vertices"])):
            if v_idx_02 in face_vertices:
                # vertex is a face vertex, therefore it exists in the merge cell 01
                for v_idx_01 in range(len(instance[cell_idx_01]["vertices"])):
                    if np.allclose(instance[cell_idx_02]["vertices"][v_idx_02],
                                   instance[cell_idx_01]["vertices"][v_idx_01]):
                        # identical vertex to v_idx_02 in cell 01 found (element at v_idx_01) --> add to mapping
                        c2_in_c1_mapping[v_idx_02] = v_idx_01
                        break
            else:
                # vertex is not a face vertex and therefore not already in merge cell 01 --> add with offset to mapping
                c2_in_c1_mapping[v_idx_02] = len(instance[cell_idx_01]["vertices"]) + idx_offset
                idx_offset += 1
        return c2_in_c1_mapping

    def _merge_volume_original(self, instance: list, cell_idx_01: int, cell_idx_02: int):
        '''
        Internal function used to merge the volume and center (original) of two given voronoi cells. Cell 01 (at idx_01)
        will contain at least two originals after the execution of this function in an np.array where each row
        represents a original.

        :param instance: The problem instance as a list containing the different voronoi cells
        :param cell_idx_01: The integer index of the first cell (this cell will be kept later)
        :param cell_idx_02: The integer index of the second cell (this cell will be deleted later)
        '''
        # update volume
        instance[cell_idx_01]["volume"] += instance[cell_idx_02]["volume"]

        # original entry will contain all originals of all merged cells
        if len(instance[cell_idx_01]["original"].shape) == 1:
            # cell 01 wasn't merged before
            instance[cell_idx_01]["original"] = np.array([instance[cell_idx_01]["original"]])
        if len(instance[cell_idx_02]["original"].shape) == 1:
            # cell 02 wasn't merged before
            instance[cell_idx_01]["original"] = np.append(instance[cell_idx_01]["original"],
                                                          [instance[cell_idx_02]["original"]], axis=0)
        else:
            # cell 02 contains at least 2 cell centers (i.e. was merged before)
            instance[cell_idx_01]["original"] = np.append(instance[cell_idx_01]["original"],
                                                          instance[cell_idx_02]["original"], axis=0)

    def _merge_adjacency_vertices(self, instance: list, cell_idx_01: int, cell_idx_02: int, face_vertices: list,
                                  c2_c1_mapping: dict):
        '''
        Internal function used by the _merge_cells function to merge adjacency lists and vertices of two given cells.
        The function first maps all vertices from cell 02 to indices in cell 01 (see _merge_c2_c1_mapping for details)
        and then combines the new adjacency entries with the existing ones for every vertex of the face where the two
        cells merged. After that, all vertices and associated adjacency lists of vertices that occur in both cells will
        be deleted from cell 02. Hence the result will only contain vertices (with associated adjacency lists) that
        don't occur in cell 01. Both vertices and adjacency lists will be then appended to cell 01.

        :param instance: The problem instance as list of voronoi cells
        :param cell_idx_01: The integer index of the first cell (this cell will be kept and expanded with the content of
            cell 02)
        :param cell_idx_02: The index of the second cell, which will be deleted piecewise (vertices, adjacency lists
            here)
        :param face_vertices: List of vertex indices from cell 02 that represent the face where the two cells got merged
        :param c2_c1_mapping: The mapping from cell 02 vertex indices to cell 01 vertex indices
        '''
        c02_adjacency = [[c2_c1_mapping[v_idx] for v_idx in adj] for adj in instance[cell_idx_02]["adjacency"]]
        # update adjacency lists of face vertices in cell 01, because they now have an additional connection
        for face_vertex_idx in face_vertices:
            tmp_adjacency = instance[cell_idx_01]["adjacency"][c2_c1_mapping[face_vertex_idx]]
            instance[cell_idx_01]["adjacency"][c2_c1_mapping[face_vertex_idx]] = list(
                np.unique(tmp_adjacency + c02_adjacency[face_vertex_idx]))

        # append adjacency data and vertices data from cell 02 to cell 01
        cell_02_vertices = instance[cell_idx_02]["vertices"]
        for idx in sorted(face_vertices, reverse=True):
            del c02_adjacency[idx]
            del cell_02_vertices[idx]
        instance[cell_idx_01]["adjacency"] += c02_adjacency
        instance[cell_idx_01]["vertices"] += cell_02_vertices

    def _merge_faces(self, instance: list, cell_idx_01: int, cell_idx_02: int, faces: list,
                     c2_c1_mapping: dict):
        '''
        Internal function used by the _merge_cells function to finish up the merge process. This function removes all
        faces to the cell 01 itself (which occurs when larger cells are merged) and appends all faces from cell 02 to
        cell 01 if the face is not a face to cell 01 (face to itself). After this, cell 02 will be deleted. To ensure
        that everything works some adjustments have to be made, i.e. decrementing every adjacent_cell entry where the
        entry is bigger than cell_idx_02. This is necessary because after the deletion of cell 02 every cell will move
        up one index. If we wouldn't decrement the indices, the indices would cause a out of bound exception or would
        not be accurate.

        :param instance: The problem instance as a list containing all voronoi cells
        :param cell_idx_01: The integer index of the first cell (that will be kept)
        :param cell_idx_02: The integer index of the second cell (which no longer exists)
        :param faces: List of vertices representing faces from cell 02 to cell 01, i.e. faces to the cell itself after
            merging
        :param c2_c1_mapping: Dict representing the mapping from cell 02 vertex indices to cell 01 vertex indices
        '''
        # delete "merge" face in cell 01 and then append all relevant faces from cell 02 to cell 01
        mapped_faces_c2 = [sorted([c2_c1_mapping[elem] for elem in face]) for face in faces]
        new_faces = []
        for face in instance[cell_idx_01]["faces"]:
            append = True
            for mapped_face in mapped_faces_c2:
                if np.array_equal(sorted(face["vertices"]), mapped_face):
                    # current face is a face to itself, because its equal to a mapped_face
                    append = False
            if append:
                # only appends if face is not equal to a mapped_face, i.e. if its not a face to itself
                new_faces.append(face)
        instance[cell_idx_01]["faces"] = new_faces

        for face in instance[cell_idx_02]["faces"]:
            # make sure to not append the "merge" face from cell 02 to the faces of cell 01
            if face["adjacent_cell"] != cell_idx_01:
                # map the vertex indices in the face to cell 01 vertex indices before appending them
                face["vertices"] = [c2_c1_mapping[elem] for elem in face["vertices"]]
                instance[cell_idx_01]["faces"].append(face)
            elif not np.any([np.array_equal(sorted([c2_c1_mapping[elem] for elem in face["vertices"]]), face_vertices)
                             for face_vertices in mapped_faces_c2]):
                face["vertices"] = [c2_c1_mapping[elem] for elem in face["vertices"]]
                instance[cell_idx_01]["faces"].append(face)

        del instance[cell_idx_02]
        # update "adjacent_cell" entry in faces of every cell where the "adjacent_cell" is greater than the cell_idx_02
        # because by deleting the cell at cell_idx_02 every following cell will move up one idx, i.e. "adjacent_cell"
        # references would be invalid or causing out of bounds exception
        for cell_idx in range(len(instance)):
            for face_idx in range(len(instance[cell_idx]["faces"])):
                if instance[cell_idx]["faces"][face_idx]["adjacent_cell"] >= 0:
                    # adjacent cell is not a boundary
                    if instance[cell_idx]["faces"][face_idx]["adjacent_cell"] == cell_idx_02:
                        # cell at cell_idx_02 joined cell at cell_idx_01, therefore adjacent_cell must be cell_idx_01
                        instance[cell_idx]["faces"][face_idx]["adjacent_cell"] = cell_idx_01
                        continue
                    if instance[cell_idx]["faces"][face_idx]["adjacent_cell"] > cell_idx_02:
                        # cell in adjacent_cell entry is still there, but the index will move up --> decrement by 1
                        instance[cell_idx]["faces"][face_idx]["adjacent_cell"] -= 1

    def _merge_cells(self, instance: list, cell_idx_01: int, cell_idx_02: int):
        '''
        Internal function used by the generate_dataset function to merge cells until there are <num_end_cells> left. The
        function merges two cells at a time, deleting the cell with the higher index in the data structure and appending
        its content to the cell with the lower index. More information on how this process works in detail can be found
        in the documentation of the used functions.

        :param instance: The problem instance in which two cells should be merged
        :param cell_idx_01: The integer index of the first merge cell
        :param cell_idx_02: The integer index of the second merge cell
        :return: The instance with the two cells merged
        '''
        # cell_idx_01 has to be smaller than cell_idx_02
        if cell_idx_01 > cell_idx_02:
            tmp = cell_idx_01
            cell_idx_01 = cell_idx_02
            cell_idx_02 = tmp

        # update volume, original
        self._merge_volume_original(instance, cell_idx_01, cell_idx_02)

        faces_cell_02 = [face["vertices"] for face in instance[cell_idx_02]["faces"] if
                         face["adjacent_cell"] == cell_idx_01]
        face_vertices_cell_02 = []
        for face in [f for f in instance[cell_idx_02]["faces"] if f["adjacent_cell"] == cell_idx_01]:
            face_vertices_cell_02 += face["vertices"]
        face_vertices_cell_02 = list(np.unique(face_vertices_cell_02))

        c2_c1_mapping = self._merge_c2_c1_mapping(instance, cell_idx_01, cell_idx_02, face_vertices_cell_02)
        # update adjacency and vertices
        self._merge_adjacency_vertices(instance, cell_idx_01, cell_idx_02, face_vertices_cell_02, c2_c1_mapping)
        # update faces
        self._merge_faces(instance, cell_idx_01, cell_idx_02, faces_cell_02, c2_c1_mapping)

        return instance

    def load_dataset(self, folder: str):
        """
        Function that loads a voronoi dataset from a folder that was previously saved with the generate_and_save_dataset
        function. Additionally the function checks some parameters with which the dataset was created and raises an
        Error if at least one parameter doesn't match

        :param folder: The folder containing the voronoi dataset in different instance (.pkl) files and a description
            (.pkl) file
        :return: 2-tuple containing the dataset (list of instances) and the description (dictionary)
        """
        # just in case the user forgot to add a / to the end of his path
        if folder[-1] != "/":
            folder += "/"

        files = os.listdir(folder)
        description_file = [name for name in files if name.__contains__("description")]
        instance_files = sorted([name for name in files if name.__contains__("instance")])

        # make sure only one description file exists
        assert len(description_file) == 1

        description = None
        description_file = description_file[0]
        with open(folder + description_file, "rb") as file:
            description = pickle.load(file)

        dataset = []
        for instance_file in instance_files:
            with open(folder + instance_file, "rb") as file:
                dataset.append(pickle.load(file))

        # test if everything is alright (tests the easy parameters for every instance)
        assert len(dataset) == description["num_instances"]
        for instance in dataset:
            assert len(instance) == description["num_end_cells"]

        return dataset, description
