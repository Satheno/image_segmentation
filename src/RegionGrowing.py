import numpy as np
import RegionGrowing3D as regro
from itertools import product
from scipy.special import comb
import pandas as pd
from tqdm import tqdm


class RegionGrowing:

    def __init__(self, inverted_grey_values, variant="voronoi", neighbor_count=6):
        """
        Region Growing object that contains a implementation for a 3d region growing algorithm for both Curve and
        Voronoi instances. Both cases, i.e. with or without inverted grey values are supported. For usage see grow
        function

        :param inverted_grey_values: Whether the problem instance was generated with inverted grey values or not
        :param variant: The variant of the algorithm that will be used. Value must be in ["voronoi", "curve"]
        :param neighbor_count: The amount of neighbors that will be examined during the region growing algorithm. Values
            must be one of the following [6, 29]
        """
        self._inverted_grey_values = inverted_grey_values
        self._variant = variant
        assert self._variant in ["voronoi", "curve"]
        self._neighbor_count = neighbor_count
        if self._inverted_grey_values:
            self._lower_threshold = 127 if self._variant == "voronoi" else -1
            self._upper_threshold = 256 if self._variant == "voronoi" else 129
        else:
            self._lower_threshold = -1 if self._variant == "voronoi" else 127
            self._upper_threshold = 128 if self._variant == "voronoi" else 256

    def grow(self, img_3d):
        """
        Region Growing function. The function will first calculate all available points that can be examined and chooses
        one of them as a seed. Then a Cython Implementation of seed based 3d region growing will be executed with this
        seed and all points in the resulting cluster will be invalidated as new seeds. The cluster itself will be saved
        in a label list, i.e. a flattened 3d array representing the instance. This process is repeated until there are
        no possible seeds left, i.e. every point is assigned to a cluster.

        :param img_3d: The problem instance as a 3d numpy array with shape (x,x,x) where each cell contains a integer
            in the range [0, 255], i.e. a 8bit value
        :return: np.array with shape (x^3,) containing the labels for every point in the 3d array
        """
        # setup calculation
        clusters = np.full(np.prod(img_3d.shape), -100)
        avail_coords = []
        dim = img_3d.shape[0]
        dim2 = dim ** 2
        mask = np.ones_like(img_3d, dtype=int)
        for coord in tqdm(list(product(list(range(img_3d.shape[0])), repeat=3)), "Region Growing Setup"):
            if self._lower_threshold < img_3d[coord[0], coord[1], coord[2]] < self._upper_threshold:
                # cell is empty, i.e. not a wall --> append coordinate to available coordinates (later used as seeds)
                avail_coords.append(coord)
            else:
                # cell is not empty, i.e. is wall or noise --> invalidate coordinate in mask
                mask[coord[0], coord[1], coord[2]] = 0

        # find clusters until there are no available coordinates (i.e. not assigned coordinates) left
        avail_coords = np.array(avail_coords)
        cluster_idx = 0
        while len(avail_coords) > 0:
            seed = avail_coords[0]

            # grower needs to be created every iteration, because <mask> changes based on its return value; uses Cython
            grower = regro.RegionGrowing3D(img_3d, mask, self._neighbor_count, self._lower_threshold,
                                           self._upper_threshold)
            region = np.where(grower.calculate_region(np.array(seed).astype(int)) == 1)

            # invalidate all coordinates in the region and remove them from available coordinates
            for del_coord in zip(region[0], region[1], region[2]):
                mask[del_coord[0], del_coord[1], del_coord[2]] = 0
                del_idx = np.where((avail_coords[:, 0] == del_coord[0]) * (avail_coords[:, 1] == del_coord[1]) * (
                        avail_coords[:, 2] == del_coord[2]))[0]
                if len(del_idx) > 0:
                    avail_coords = np.delete(avail_coords, del_idx[0], 0)
                    # calculate idx of element in 3d flattened array and set the cluster_idx
                    clusters[dim2 * del_coord[0] + dim * del_coord[1] + del_coord[2]] = cluster_idx
            # cluster calculation is done now --> increment cluster_idx
            cluster_idx += 1

        return clusters

    def _rand_index_score(self, clustering, clustering_truth):
        """
        Helper function used to calculate rands index for the clustering
        Code from https://stackoverflow.com/questions/49586742/rand-index-function-clustering-performance-evaluation
        because my own code would be slower, especially for larger datasets

        :param clustering: np.array with shape (num_points,) containing the guessed integer labels for the points
        :param clustering_truth: np.array with shape (num_points,) containing the ground truth integer labels for the
            points
        :return: Rands index
        """
        tp_plus_fp = comb(np.bincount(clustering), 2).sum()
        tp_plus_fn = comb(np.bincount(clustering_truth), 2).sum()
        A = np.c_[(clustering, clustering_truth)]
        tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum() for i in set(clustering))
        fp = tp_plus_fp - tp
        fn = tp_plus_fn - tp
        tn = comb(len(A), 2) - tp - fp - fn
        return (tp + tn) / (tp + fp + fn + tn)

    def _calculate_label_mapping(self, clustering, clustering_truth):
        """
        Helper function used by both evaluation metrics. The function maps the cluster labels obtained by the region
        growing algorithm to the labels in the ground truth.Necessary because the calculated labels from the region
        growing algorithm won't match the labels from the ground truth. This is due to the different iteration order in
        both region growing algorithm and ground truth generation process. Example:
        Ground truth:          [2, 2, 2, 1, 3, 3, 0, 0, 0]
        RG Algorithm:          [1, 1, 4, 0, 2, 4, 3, 5, 5]
        RG Algorithm mapped:   [2, 2, 4, 1, 3, 4, 0, 5, 5]

        :param clustering: np.array of shape (x,) containing the labels only for valid points, i.e. no labels for
            boundary points
        :param clustering_truth: np.array of shape (x,) containing the true labels only for the valid points, i.e. no
            labels for boundary points
        :return: dict representing the mapping from RG algorithm labels to ground truth labels (key: RG label, value:
            ground truth label)
        """
        mapping = {}
        mapping_offset = 0
        truth_uniques = np.unique(clustering_truth)
        for idx, elem in enumerate(tqdm(clustering_truth, "Adapting Ground Truth Label Schema")):
            truth_elem_idxs = np.where(clustering_truth == elem)[0]
            mapping_uniques = np.unique(clustering[truth_elem_idxs])
            if len(mapping_uniques) == 1:
                # clustering at all truth_elem_idxs has the same value --> clear mapping possible
                mapping[clustering[truth_elem_idxs[0]]] = elem
            else:
                # clustering at all truth_elem_idxs has at least two different values --> no clear mapping possible
                # TODO: Decide what to do in this case --> currently first occurrence is mapped (rest left as is)
                for mapping_idx, mapping_elem in enumerate(mapping_uniques):
                    if mapping_elem in mapping.keys():
                        continue
                    if mapping_idx == 0:
                        mapping[mapping_elem] = elem
                    else:
                        mapping[mapping_elem] = len(truth_uniques) + mapping_offset
                        mapping_offset += 1
        return mapping

    def _information_variation_score(self, clustering, clustering_truth):
        """
        Helper function used to calculate the variation of information.

        :param clustering: np.array with shape (num_points,) containing the guessed integer labels for the points
        :param clustering_truth: np.array with shape (num_points,) containing the ground truth integer labels for the
            points
        :return:
        """
        clustering = [np.zeros(count) for count in clustering.value_counts()]
        clustering_truth = [np.zeros(count) for count in clustering_truth.value_counts()]

        n = float(sum([len(x) for x in clustering]))
        sigma = 0.0
        for guess_cluster in clustering:
            p = len(guess_cluster) / n
            for truth_cluster in clustering_truth:
                q = len(truth_cluster) / n
                r = len(set(guess_cluster) & set(truth_cluster)) / n
                if r > 0.:
                    sigma += r * (np.log2(r / p) + np.log2(r / q))
        return abs(sigma)

    def _preprocess_labels(self, clustering, clustering_truth, removed_boundary=False, adapted_labels=False):
        """
        Helper function used to preprocess the the label np.arrays. The function removes all boundary labels, i.e.
        entries with value -100 and maps the calculated RG labels to the ground truth labels (see
        _calculate_label_mapping for details)

        :param clustering: np.array with shape (num_points,) containing the guessed integer labels for the points
        :param clustering_truth: np.array with shape (num_points,) containing the ground truth integer labels for the
            points
        :param removed_boundary: boolean that indicates whether all boundary labels have been removed or not
        :param adapted_labels: boolean that indicates whether the RG calculated labels have been mapped to ground truth
            labels or not
        :return: preprocessed RG calculated labels and ground truth labels
        """
        if not removed_boundary:
            clustering = np.delete(clustering, np.where(clustering == -100))
            clustering_truth = np.delete(clustering_truth, np.where(clustering_truth == -100))
        if not adapted_labels:
            mapping = self._calculate_label_mapping(clustering, clustering_truth)
            clustering = np.array([mapping[elem] for elem in clustering])
        return clustering, clustering_truth

    def calculate_rands_index(self, clustering, clustering_truth, removed_boundary=False, adapted_labels=False):
        """
        Function that calculates rands index after preprocessing steps (if necessary). If you want to calculate both
        evaluation metric use the evaluate function, because this will be faster.

        :param clustering: np.array with shape (num_points,) containing the guessed integer labels for the points
        :param clustering_truth: np.array with shape (num_points,) containing the ground truth integer labels for the
            points
        :param removed_boundary: boolean that indicates whether all boundary labels have been removed or not
        :param adapted_labels: boolean that indicates whether the RG calculated labels have been mapped to ground truth
            labels or not
        :return: rands index
        """
        assert clustering_truth.shape == clustering.shape
        if not np.any([removed_boundary, adapted_labels]):
            clustering, clustering_truth = self._preprocess_labels(clustering, clustering_truth, removed_boundary,
                                                                   adapted_labels)
        return self._rand_index_score(clustering, clustering_truth)

    def calculate_information_variation(self, clustering, clustering_truth, removed_boundary=False,
                                        adapted_labels=False):
        """
        Function that calculates the variation of information after preprocessing steps (if necessary). If you want to
        calculate both evaluation metric use the evaluate function, because this will be faster.

        :param clustering: np.array with shape (num_points,) containing the guessed integer labels for the points
        :param clustering_truth: np.array with shape (num_points,) containing the ground truth integer labels for the
            points
        :param removed_boundary: boolean that indicates whether all boundary labels have been removed or not
        :param adapted_labels: boolean that indicates whether the RG calculated labels have been mapped to ground truth
            labels or not
        :return: variation of information
        """
        assert clustering_truth.shape == clustering.shape
        if not np.any([removed_boundary, adapted_labels]):
            clustering, clustering_truth = self._preprocess_labels(clustering, clustering_truth, removed_boundary,
                                                                   adapted_labels)
        return self._information_variation_score(pd.Series(clustering), pd.Series(clustering_truth))

    def evaluate(self, clustering, clustering_truth):
        """
        Function that calculates both rands index and the variation of information after preprocessing steps.

        :param clustering: np.array with shape (num_points,) containing the guessed integer labels for the points
        :param clustering_truth: np.array with shape (num_points,) containing the ground truth integer labels for the
            points
        :return: 2-tuple containing rands index and variation of information
        """
        assert clustering_truth.shape == clustering.shape
        clustering, clustering_truth = self._preprocess_labels(clustering, clustering_truth)
        rand_index = self.calculate_rands_index(clustering, clustering_truth, True, True)
        information_variance = self.calculate_information_variation(clustering, clustering_truth, True, True)
        return rand_index, information_variance
