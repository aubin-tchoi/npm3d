"""
Computation of neighborhood on a point cloud.
"""
import sys
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from sklearn.neighbors import KDTree

from ply import read_ply


def brute_force_spherical(queries, supports, radius: float):
    """
    Brute force computation of a spherical neighborhood associated with each query.

    Returns:
        The neighborhoods.
    """
    return [
        supports[np.linalg.norm(supports - query, axis=1) < radius] for query in queries
    ]


def brute_force_KNN(queries, supports, k: int):
    """
    Brute force computation of k nearest neighbors for each query.

    Returns:
        The neighborhoods.
    """
    neighborhoods = []
    for query in queries:
        distances = np.linalg.norm(supports - query, axis=1)
        neighborhoods.append(supports[np.argpartition(distances, k)])

    return neighborhoods


def kdtree_spherical(tree, queries, radius: float) -> float:
    """
    Computes a spherical neighborhood associated with each query using a provided KDTree.

    Returns:
        The computation time.
    """
    time_ref = time.time()
    tree.query_radius(queries, radius)
    return time.time() - time_ref


def compare_inference_build_times(
    runtimes: List[float],
    build_times: List[float],
    leaf_sizes: List[int],
) -> None:
    """
    Plots the inference and build times for various leaf sizes.
    """
    sum_times = [runtimes[i] + build_times[i] for i in range(len(leaf_sizes))]
    print(
        f"Optimal leaf size when taking into account the build time: {leaf_sizes[np.argmin(sum_times)]}"
    )
    plt.figure()
    plt.plot(leaf_sizes, runtimes, label="inference time")
    plt.plot(leaf_sizes, build_times, label="build time")
    plt.plot(
        leaf_sizes,
        sum_times,
        label="sum",
    )

    plt.xlabel("leaf size")
    plt.ylabel("time")
    plt.legend()
    plt.show()


def find_farthest_points(points):
    """
    Finds the two farthest points apart in the point cloud using its convex hull.
    Additionally, prints the distance between them.
    """
    # computing the convex hull of the set of points
    convex_hull = ConvexHull(points)
    points_on_hull = points[convex_hull.vertices, :]

    # naive way of finding the best pair in O(hull_size ** 2)
    hdist = cdist(points_on_hull, points_on_hull, metric="euclidean")

    # getting the farthest apart points
    bestpair = np.unravel_index(hdist.argmax(), hdist.shape)

    dist = np.linalg.norm(points_on_hull[bestpair[0]] - points_on_hull[bestpair[1]])
    print(f"Distance between the two farthest apart points: {dist:.2f}.")

    return points_on_hull[bestpair[0]], points_on_hull[bestpair[1]]


if __name__ == "__main__":

    # loading the point cloud
    file_path = "../data/indoor_scan.ply"
    data = read_ply(file_path)

    # concatenating the data
    points = np.vstack((data["x"], data["y"], data["z"])).T

    use_brute_force = sys.argv[1] == "brute_force" if len(sys.argv) > 1 else False

    # brute force neighborhoods
    if use_brute_force:

        # defining the search parameters
        neighbors_num = 100
        radius = 0.2
        num_queries = 10

        # picking random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        # spherical search
        t0 = time.time()
        brute_force_spherical(queries, points, radius)
        t1 = time.time()

        # KNN search
        brute_force_KNN(queries, points, neighbors_num)
        t2 = time.time()

        # printing the timing results
        print(
            f"{num_queries:d} spherical neighborhoods computed in {t1 - t0:.3f} seconds"
        )
        print(f"{num_queries:d} KNN computed in {t2 - t1:.3f} seconds")

        # estimating the time required to compute all neighborhoods in the cloud
        total_spherical_time = points.shape[0] * (t1 - t0) / num_queries
        total_KNN_time = points.shape[0] * (t2 - t1) / num_queries
        print(
            f"Computing spherical neighborhoods on whole cloud : {total_spherical_time / 3600:.0f} hours"
        )
        print(
            "Computing KNN on whole cloud : {:.0f} hours".format(total_KNN_time / 3600)
        )

    # KDTree neighborhoods
    else:
        leaf_sizes = [61]
        num_queries = 1000

        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        runtimes, build_times = [], []
        radius = 0.2

        for leaf_size in leaf_sizes:
            t_ref = time.time()
            kdtree = KDTree(points, leaf_size=leaf_size)
            build_times.append(time.time() - t_ref)

            runtime = kdtree_spherical(kdtree, queries, radius)
            print(
                f"{num_queries} spherical neighborhoods computed in {runtime:.3f} seconds"
            )
            runtimes.append(runtime)

        optimal_leaf_size = leaf_sizes[np.argmin(runtimes)]
        print(f"Optimal leaf size: {optimal_leaf_size}")
        compare_inference_build_times(runtimes, build_times, leaf_sizes)

        plt.figure()
        kdtree = KDTree(points, leaf_size=optimal_leaf_size)

        radii = [1e-2, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2.5, 5, 7.5, 10, 12.5, 15]
        runtime = np.zeros(len(radii))
        for idx, radius in enumerate(radii):
            runtime[idx] = kdtree_spherical(kdtree, queries, radius)

        print(
            f"Computing spherical neighborhoods on whole cloud: {points.shape[0] * runtime[2] / num_queries:.2f} s."
        )
        plt.plot(radii, runtime, label=f"leaf size: {optimal_leaf_size}")

        plt.xlabel("radius")
        plt.ylabel("execution time")
        plt.legend()
        plt.show()
