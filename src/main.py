import sys

if not sys.path.__contains__("D:\\Projects\\Python\\image_segmentation"):
    sys.path.append("D:\\Projects\\Python\\image_segmentation")
import numpy as np
from src.DatasetGenerator import CurveDatasetGenerator, VoronoiDatasetGenerator
from src.Visualizer import CurveVisualizer, VoronoiVisualizer
from src.RegionGrowing import RegionGrowing

if __name__ == '__main__':
    # Data selection and inverted grey values setting
    curves = True
    voronoi = False
    inverted_grey_values = True

    if curves:
        # Curve Generation
        gen = CurveDatasetGenerator(3, 10, 80, 4, .2, 20, 842)
        dataset, description = gen.load_dataset("../datasets/test_set/")
        # Curve visualization
        cur_vis = CurveVisualizer(64, True, inverted_grey_values, alpha_voxels=False)
        cur_vis.draw_original(dataset[0])
        eight_bit, clustering_truth = cur_vis.draw_pixel(dataset[0])

    if voronoi:
        # Voronoi Generation
        gen = VoronoiDatasetGenerator(1, 100, 10)
        ds, desc = gen.generate_dataset()
        # Voronoi visualization
        vor_vis = VoronoiVisualizer(50, inverted_grey_values, max_dist=0.02, alpha_voxels=False)
        vor_vis.draw_original(ds[0])
        eight_bit, clustering_truth = vor_vis.draw_pixel(ds[0])

    if curves != voronoi:
        # Algorithm (3D region growing) execution
        grower = RegionGrowing(inverted_grey_values, variant="voronoi" if voronoi else "curve")
        clustering = grower.grow(eight_bit.astype(int))

        # Evaluation of the resulting clustering
        rands_index, information_variance = grower.check_results(clustering, clustering_truth)
        print(f"Rands Index: {rands_index}\nInformation Variation: {information_variance}")

    print("IDE Debug Breakpoint")
