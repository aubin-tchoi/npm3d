"""
Implementation of the RANSAC algorithm for plane detection.
"""
import time
import warnings
from typing import Tuple, Optional

import numpy as np
from sklearn.neighbors import KDTree

from ply import write_ply, read_ply
from sampling import (
    random_sample,
    fast_random_sample,
    local_sample,
    fast_local_sample,
)
from utils import timeit, parse_args, compute_normals

warnings.filterwarnings("ignore")


def compute_plane(points: np.ndarray):
    """
    Computes a plane (reference point + normal) from the given point cloud.

    If np.linalg.norm(normal) is too small we will get NaN values,
    which is not an issue because n_points_in_plane will be equal to 0.
    """
    point = points[0].reshape((3, 1))
    normal = np.cross(points[1] - point.T, points[2] - point.T).reshape((3, 1))

    return point, normal / np.linalg.norm(normal)


def in_plane(
    points: np.ndarray,
    ref_pt: np.ndarray,
    normal: np.ndarray,
    threshold_in: float = 0.1,
    normals: Optional[np.ndarray] = None,
    max_angle: float = 0.1,
):
    """
    Finds the indices of the points on the given plane in a point cloud.
    """
    dists = np.abs((points - ref_pt.T) @ normal)
    indices = (dists < threshold_in).squeeze()
    if normals is not None:
        indices = np.logical_and(indices, aligned_normals(normals, normal, max_angle))

    return indices


def aligned_normals(normals: np.ndarray, ref_direction: np.ndarray, max_angle: float):
    """
    Finds the indices of the points whose normals are aligned with a given vector in a point cloud.
    """
    angles = np.arccos(
        np.clip(
            (normals @ ref_direction).squeeze(), -1, 1
        )  # clipping in case the angle is superior to 1 by an epsilon
    )  # absolute value of the angle between two vectors

    return angles < max_angle


def RANSAC(
    points: np.ndarray,
    nb_draws: int,
    threshold_in: float = 0.1,
    normals: np.ndarray = None,
    max_angle: float = 0.1,
    sampling_method: str = "fast_local",
    prob_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Implementation of the RANSAC algorithm.
    """

    best_vote = 3
    best_pt_plane = np.zeros((3, 1))
    best_normal_plane = np.zeros((3, 1))
    kd_tree = None
    if sampling_method.lower().endswith("local"):
        kd_tree = KDTree(points)

    for i in range(nb_draws):
        # selecting a random set of 3 points
        random_points = (
            fast_local_sample(points, kd_tree)
            if sampling_method == "fast_local"
            else local_sample(points, kd_tree)
            if sampling_method == "local"
            else random_sample(points)
            if sampling_method == "basic"
            else fast_random_sample(points)
        )

        # evaluating the random planes
        sample_ref, sample_normal = compute_plane(random_points)
        n_points_in_plane = in_plane(
            points, sample_ref, sample_normal, threshold_in, normals, max_angle
        ).sum()

        if n_points_in_plane > best_vote:
            best_pt_plane = sample_ref
            best_normal_plane = sample_normal
            best_vote = n_points_in_plane

        # early stopping criterion
        if 1 - (1 - (best_vote / len(points)) ** 3) ** (i + 1) > prob_threshold:
            print("Early stopping.")
            break

    return best_pt_plane, best_normal_plane, best_vote


@timeit
def recursive_RANSAC(
    points: np.ndarray,
    nb_draws: int = 100,
    threshold_in: float = 0.5,
    nb_planes: int = 2,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs the RANSAC algorithm iteratively to find the "best plane" among all the points and then the "second-best plane"
    among the remaining points, and so on until nb_planes planes are found.
    """

    n_points = len(points)
    plane_indices = np.arange(0, 0)
    plane_labels = np.arange(0, 0)
    remaining_indices = np.arange(n_points)

    for label in range(nb_planes):
        # finding the plane that contains the most points
        ref, normal, _ = RANSAC(points[remaining_indices], nb_draws, threshold_in)
        pts_in_plane = in_plane(points[remaining_indices], ref, normal, threshold_in)

        if verbose:
            print(f"Reference point of the {label + 1}-th plane: ", end="")
            print(ref.squeeze())
            print(f"Normal of the {label + 1}-th plane: ", end="")
            print(normal.squeeze(), end="\n\n")

        # updating the indexes and labels
        plane_indices = np.append(plane_indices, remaining_indices[pts_in_plane])
        plane_labels = np.append(plane_labels, np.repeat(label, pts_in_plane.sum()))
        remaining_indices = remaining_indices[~pts_in_plane]

    return plane_indices, remaining_indices, plane_labels


@timeit
def recursive_RANSAC_with_normals(
    points: np.ndarray,
    nb_draws: int = 100,
    threshold_in: float = 0.5,
    nb_planes: int = 2,
    max_angle: float = 0.1,
    normals: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs the RANSAC algorithm iteratively to find the "best plane" among all the points and then the "second-best plane"
    among the remaining points, and so on until nb_planes planes are found.
    Adaptation of the previous function with an additional condition to include a point in a plane: it also needs its
    normal to form an angle small enough with the normal to the plane.
    """

    n_points = len(points)
    plane_indices = np.arange(0, 0)
    plane_labels = np.arange(0, 0)
    remaining_indices = np.arange(n_points)
    if normals is None:
        normals = compute_normals(points, points, radius=0.6)

    for label in range(nb_planes):
        ref, normal, _ = RANSAC(
            points[remaining_indices],
            nb_draws,
            threshold_in,
            normals[remaining_indices],
            max_angle,
        )
        pts_in_plane = in_plane(
            points[remaining_indices],
            ref,
            normal,
            threshold_in,
            normals[remaining_indices],
            max_angle,
        )

        plane_indices = np.append(plane_indices, remaining_indices[pts_in_plane])
        plane_labels = np.append(plane_labels, np.repeat(label, pts_in_plane.sum()))
        remaining_indices = remaining_indices[~pts_in_plane]
        print(f"{label + 1}-th plane found.")

    return plane_indices, remaining_indices, plane_labels


if __name__ == "__main__":

    args = parse_args()

    # loading the point cloud
    data = read_ply(args.file_path)
    point_cloud = np.vstack((data["x"], data["y"], data["z"])).T
    # dealing with incomplete ply files by filling with zeros
    try:
        colors = np.vstack((data["red"], data["green"], data["blue"])).T
    except ValueError:
        colors = np.zeros(point_cloud.shape)
    try:
        labels = data["label"]
    except ValueError:
        labels = np.zeros((point_cloud.shape[0], 1))

    # computes the plane passing through 3 randomly chosen points
    print("\n--- 1) and 2) ---\n")

    pts = point_cloud[np.random.randint(0, len(point_cloud), size=3)]
    t0 = time.time()
    pt_plane, normal_plane = compute_plane(pts)
    t1 = time.time()
    print(f"Plane computation carried out in {t1 - t0:.3f} seconds")

    t0 = time.time()
    points_in_plane = in_plane(point_cloud, pt_plane, normal_plane, args.threshold_in)
    t1 = time.time()
    print(f"Plane extraction carried out in {t1 - t0:.3f} seconds")
    pts_in_plane_indices = points_in_plane.nonzero()[0]
    remaining_pts_indices = (1 - points_in_plane).nonzero()[0]

    write_ply(
        "../plane.ply",
        [point_cloud[pts_in_plane_indices], colors[pts_in_plane_indices], labels[pts_in_plane_indices]],
        ["x", "y", "z", "red", "green", "blue", "label"],
    )
    write_ply(
        "../remaining_points_plane.ply",
        [
            point_cloud[remaining_pts_indices],
            colors[remaining_pts_indices],
            labels[remaining_pts_indices],
        ],
        ["x", "y", "z", "red", "green", "blue", "label"],
    )

    # computing the plane that fits best the point cloud
    print("\n--- 3) ---\n")

    t0 = time.time()
    best_plane_point, best_plane_normal, _ = RANSAC(
        point_cloud, args.nb_draws, args.threshold_in
    )
    t1 = time.time()
    print(f"RANSAC done in {t1 - t0:.3f} seconds")

    points_in_plane = in_plane(
        point_cloud, best_plane_point, best_plane_normal, args.threshold_in
    )
    pts_in_plane_indices = points_in_plane.nonzero()[0]
    remaining_pts_indices = (1 - points_in_plane).nonzero()[0]

    write_ply(
        "../best_plane.ply",
        [point_cloud[pts_in_plane_indices], colors[pts_in_plane_indices], labels[pts_in_plane_indices]],
        ["x", "y", "z", "red", "green", "blue", "label"],
    )
    write_ply(
        "../remaining_points_best_plane_.ply",
        [
            point_cloud[remaining_pts_indices],
            colors[remaining_pts_indices],
            labels[remaining_pts_indices],
        ],
        ["x", "y", "z", "red", "green", "blue", "label"],
    )

    # finding "all the planes" in the cloud
    print("\n--- 4) ---\n")

    pts_in_plane_indices, remaining_pts_indices, pts_in_plane_labels = recursive_RANSAC(
        point_cloud, args.nb_draws, args.threshold_in, args.nb_planes
    )
    write_ply(
        f"../best_planes_{str(args.threshold_in).replace('.', '-')}.ply",
        [
            point_cloud[pts_in_plane_indices],
            colors[pts_in_plane_indices],
            labels[pts_in_plane_indices],
            pts_in_plane_labels.astype(np.int32),
        ],
        ["x", "y", "z", "red", "green", "blue", "label", "plane_label"],
    )
    write_ply(
        f"../remaining_points_best_planes_{str(args.threshold_in).replace('.', '-')}.ply",
        [
            point_cloud[remaining_pts_indices],
            colors[remaining_pts_indices],
            labels[remaining_pts_indices],
        ],
        ["x", "y", "z", "red", "green", "blue", "label"],
    )

    # improved implementation
    print("\n--- 5) ---\n")

    pts_in_plane_indices, remaining_pts_indices, pts_in_plane_labels = recursive_RANSAC_with_normals(
        point_cloud,
        args.nb_draws,
        args.threshold_in,
        args.nb_planes,
        args.max_angle,
    )
    write_ply(
        "../best_planes_normals.ply",
        [
            point_cloud[pts_in_plane_indices],
            colors[pts_in_plane_indices],
            labels[pts_in_plane_indices],
            pts_in_plane_labels.astype(np.int32),
        ],
        ["x", "y", "z", "red", "green", "blue", "label", "plane_label"],
    )
    write_ply(
        "../remaining_points_best_planes_normals.ply",
        [
            point_cloud[remaining_pts_indices],
            colors[remaining_pts_indices],
            labels[remaining_pts_indices],
        ],
        ["x", "y", "z", "red", "green", "blue", "label"],
    )

    print("Done!")
