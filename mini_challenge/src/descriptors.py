"""
Implementation of local PCA on point clouds for normals computations and feature extraction (PCA-based descriptors).
"""
import argparse
from typing import Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree

from .ply import write_ply, read_ply


def PCA(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the eigenvalues and eigenvectors of the covariance matrix of a point cloud.
    """
    barycenter = points.mean(axis=0)
    centered_points = points - barycenter
    cov_matrix = centered_points.T @ centered_points / points.shape[0]

    return np.linalg.eigh(cov_matrix)


def compute_local_PCA(
    query_points: np.ndarray,
    cloud_points: np.ndarray,
    nghbrd_search: str = "spherical",
    radius: Optional[float] = None,
    k: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes PCA on the neighborhoods of all query_points in cloud_points.

    Returns:
        all_eigenvalues: (N, 3)-array of the eigenvalues associated with each query point.
        all_eigenvectors: (N, 3, 3)-array of the eigenvectors associated with each query point.
    """

    kdtree = KDTree(cloud_points)
    neighborhoods = (
        kdtree.query_radius(query_points, radius)
        if nghbrd_search.lower() == "spherical"
        else kdtree.query(query_points, k=k, return_distance=False)
        if nghbrd_search.lower() == "knn"
        else None
    )

    # checking the sizes of the neighborhoods and plotting the histogram
    if nghbrd_search.lower() == "spherical":
        neighborhood_sizes = [neighborhood.shape[0] for neighborhood in neighborhoods]
        print(
            f"Average size of neighborhoods: {np.mean(neighborhood_sizes):.4f}\n"
            f"Standard deviation: {np.std(neighborhood_sizes):.4f}\n"
            f"Min: {np.min(neighborhood_sizes)}, max: {np.max(neighborhood_sizes)}\n"
        )
        hist_values, _, __ = plt.hist(neighborhood_sizes, bins="auto")
        plt.title(
            f"Histogram of the neighborhood sizes for {hist_values.shape[0]} bins"
        )
        plt.xlabel("Neighborhood size")
        plt.ylabel("Number of neighborhoods")
        plt.show()

    all_eigenvalues = np.zeros((query_points.shape[0], 3))
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))

    for i, point in enumerate(query_points):
        all_eigenvalues[i], all_eigenvectors[i] = PCA(cloud_points[neighborhoods[i]])

    return all_eigenvalues, all_eigenvectors


def compute_features(
    query_points: np.ndarray, cloud_points: np.ndarray, radius: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes PCA-based descriptors on a point cloud.
    """
    all_eigenvalues, all_eigenvectors = compute_local_PCA(
        query_points, cloud_points, radius=radius
    )
    lbd3, lbd2, lbd1 = (
        all_eigenvalues[:, 0],
        all_eigenvalues[:, 1],
        all_eigenvalues[:, 2],
    )
    lbd1 += 1e-6

    normals = all_eigenvectors[:, :, 0]

    verticality = 2 * np.arcsin(np.abs(normals[:, 2])) / np.pi
    linearity = 1 - lbd2 / lbd1
    planarity = (lbd2 - lbd3) / lbd1
    sphericity = lbd3 / lbd1

    return verticality, linearity, planarity, sphericity


def parse_args() -> argparse.Namespace:
    """
    Parses the command line arguments. Also produces the help message.
    """
    parser = argparse.ArgumentParser(description="Launches runs of PCA computation")

    parser.add_argument(
        "--radius",
        type=float,
        default=0.5,
        help="Radius of the spherical search",
    )
    parser.add_argument(
        "--k", type=int, default=30, help="Number of neighbors in the KNN search"
    )
    parser.add_argument(
        "--skip_pca_check",
        action="store_false",
        help="Skip the check on PCA computation",
    )
    parser.add_argument(
        "--skip_normals",
        action="store_false",
        help="Skip normals computation",
    )
    parser.add_argument(
        "--skip_descriptors",
        action="store_false",
        help="Skip descriptors computation",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # loading the cloud as a [N x 3] matrix
    cloud_path = "../data/Lille_street_small.ply"
    cloud_ply = read_ply(cloud_path)
    cloud = np.vstack((cloud_ply["x"], cloud_ply["y"], cloud_ply["z"])).T

    if not args.skip_pca_check:
        eigenvalues, eigenvectors = PCA(cloud)
        assert np.allclose(eigenvalues, [5.25050177, 21.7893201, 89.58924003])

    if not args.skip_normals:
        # spherical neighborhoods
        sph_normals = compute_local_PCA(cloud, cloud, radius=0.5)[1][:, :, 0]
        write_ply(
            "../Lille_street_small_normals.ply",
            (cloud, sph_normals),
            ["x", "y", "z", "nx", "ny", "nz"],
        )

        # knn neighborhoods
        knn_normals = compute_local_PCA(cloud, cloud, nghbrd_search="knn", k=30)[1][
            :, :, 0
        ]
        write_ply(
            "../Lille_street_small_normals_knn.ply",
            (cloud, knn_normals),
            ["x", "y", "z", "nx", "ny", "nz"],
        )

    if not args.skip_descriptors:
        verticality, linearity, planarity, sphericity = compute_features(
            cloud, cloud, 0.5
        )
        write_ply(
            "../Lille_street_small_normals_feats.ply",
            [cloud, verticality, linearity, planarity, sphericity],
            ["x", "y", "z", "verticality", "linearity", "planarity", "sphericity"],
        )
