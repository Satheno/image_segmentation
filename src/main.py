import sys

if not sys.path.__contains__("D:\\Projects\\Python\\image_segmentation"):
    sys.path.append("D:\\Projects\\Python\\image_segmentation")
import numpy as np
import matplotlib.pyplot as plt
from src.DatasetGenerator import CurveDatasetGenerator, VoronoiDatasetGenerator
from src.Visualizer import CurveVisualizer, VoronoiVisualizer

if __name__ == '__main__':
    # vor = pv.compute_voronoi([[.1, .1, .1], [.9, .9, .9]], [[0, 1], [0, 1], [0, 1]], .2)

    # gen = CurveDatasetGenerator(3, 10, 80, 4, .2, 20, 842)
    # gen.generate_and_save_dataset("../datasets/test_set/")
    # dataset, description = gen.load_dataset("../datasets/test_set/")
    # vis = CurveVisualizer(16, True, True)
    # vis.draw_original(dataset[0])
    # vis.draw_pixel(dataset[0])
    gen = VoronoiDatasetGenerator(1, 100, 15)
    ds, desc = gen.generate_dataset()
    vis = VoronoiVisualizer(50, max_dist=0.02, inverted_grey_values=False, alpha_voxels=False)
    # vis.draw_original(ds[0])
    vis.draw_pixel(ds[0])

    print("IDE Debug Breakpoint")
