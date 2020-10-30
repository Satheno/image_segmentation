import numpy as np
import splipy as sp
import random
import pickle
import os
import splipy.curve_factory as curve_factory
from scipy.spatial.distance import cdist


class DatasetGenerator:

    def __init__(self, num_instances: int, seed=42):
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
        pass

    def load_dataset(self, folder: str):
        pass


class CurveDatasetGenerator(DatasetGenerator):

    def __init__(self, num_instances: int, num_curves: int, num_eval_points: int, num_control_points: int,
                 min_distance: float, min_length: int, seed=42):
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
        :param min_length: The minimal amount of points of the eval_points that must be located inside the unit cube
        :param min_distance: The minimal distance between every pair of curves
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

    def __init__(self, num_instances: int):
        super().__init__(num_instances)
        # TODO: Implement this class

    def _generate_instance(self):
        pass
