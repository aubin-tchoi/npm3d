import argparse
from functools import wraps
from time import perf_counter
from typing import Optional, Tuple, Callable

import numpy as np
from sklearn.neighbors import KDTree


def PCA(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the eigenvalues and eigenvectors of the covariance matrix of a point cloud.
    """
    barycenter = points.mean(axis=0)
    centered_points = points - barycenter
    cov_matrix = centered_points.T @ centered_points / points.shape[0]

    return np.linalg.eigh(cov_matrix)


def compute_normals(
    query_points: np.ndarray,
    cloud_points: np.ndarray,
    nghbrd_search: str = "spherical",
    radius: Optional[float] = None,
    k: Optional[int] = None,
) -> np.ndarray:
    """
    Computes a local normals for every query point using a PCA on their neighborhoods.

    Returns:
        utils: (N, 3)-array of the normals associated with each query point.
    """

    kdtree = KDTree(cloud_points)

    neighborhoods = (
        kdtree.query_radius(query_points, radius)
        if nghbrd_search.lower() == "spherical"
        else kdtree.query(query_points, k=k, return_distance=False)
        if nghbrd_search.lower() == "knn"
        else None
    )

    normals = np.zeros((query_points.shape[0], 3))

    for i in range(len(query_points)):
        normals[i] = PCA(cloud_points[neighborhoods[i]])[1][:, 0]

    return normals


def grid_subsampling(points, voxel_size: float):
    """
    Performs a voxel subsampling on the point cloud.
    """
    non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(
        ((points - np.min(points, axis=0)) // voxel_size).astype(int),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    idx_pts_vox_sorted = np.argsort(inverse)

    subsampled_points = np.zeros((len(non_empty_voxel_keys), 3), points.dtype)

    last_seen = 0
    for idx in range(len(non_empty_voxel_keys)):
        indexes_in_voxel = idx_pts_vox_sorted[
            last_seen : last_seen + nb_pts_per_voxel[idx]
        ]
        # barycenter of points
        subsampled_points[idx] = points[indexes_in_voxel].mean(axis=0)

        last_seen += nb_pts_per_voxel[idx]

    return subsampled_points


def parse_args() -> argparse.Namespace:
    """
    Parses the command line arguments. Also produces the help message.
    """
    parser = argparse.ArgumentParser(description="Surface reconstruction")

    parser.add_argument(
        "--file_path",
        type=str,
        default="../data/indoor_scan.ply",
        help="Path to the ply file",
    )
    parser.add_argument(
        "--nb_draws",
        type=int,
        default=100,
        help="Number of draws in the RANSAC algorithm",
    )
    parser.add_argument(
        "--threshold_in",
        type=float,
        default=0.2,
        help="Threshold on distance to plane in the RANSAC algorithm",
    )
    parser.add_argument(
        "--max_angle",
        type=float,
        default=0.1,
        help="Threshold on the angle between normals in the RANSAC algorithm",
    )
    parser.add_argument(
        "--nb_planes",
        type=int,
        default=10,
        help="Number of planes to detect using the RANSAC algorithm",
    )

    return parser.parse_args()


def timeit(func: Callable) -> Callable:
    """
    Decorator for timing function execution time.
    """

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} took {total_time:.2f} seconds")
        return result

    return timeit_wrapper
