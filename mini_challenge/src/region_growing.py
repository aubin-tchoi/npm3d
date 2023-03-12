from queue import Queue
from typing import Union, List, Optional, Tuple, Dict

import numpy as np
from sklearn.neighbors import KDTree

from .perf_monitoring import timeit

default_thresholds = {
    "omnivariance": 20,
    "planarity": 65,
    "neighborhood_size": 300,
    "moment_x_sq": 400,
    "moment_y_sq": 400,
}


def region_criterion(
    *features_comparison: Tuple[Union[np.ndarray, float], float, float],
) -> np.ndarray:
    """
    Adds a condition that is checked when adding points to a region.

    Args:
        features_comparison: A list of tuples (feat, feat_ref, feat_threshold) where feat contains the features of the
        candidate points, feat_ref a reference value and feat_threshold the maximum distance between feat and feat_ref.
    Returns:
         in_region: list of the indices of the points to add.
    """

    conditions = np.zeros(
        (len(features_comparison), features_comparison[0][0].shape[0])
    )
    for cond, (feat, feat_ref, feat_threshold) in enumerate(features_comparison):
        conditions[cond] = np.abs(feat - feat_ref) < feat_threshold

    return np.all(
        conditions,
        axis=0,
    )


def select_seed(point_cloud: np.ndarray) -> Union[np.ndarray, int]:
    """
    Selects a seed in a point cloud.
    The current implementation returns the first point in the cloud. Since the order of the points is arbitrary, this
    should come down to the same thing as taking a random point, albeit faster.
    """
    return 0


def get_avg_criterion(
    feature: np.ndarray,
    candidate: Union[np.ndarray, int],
    cluster: np.ndarray,
    threshold: float,
) -> Tuple[Union[np.ndarray, float], float, float]:
    return (
        feature[candidate],
        feature[cluster].mean(),
        threshold * np.clip(feature[cluster].std(), a_min=1, a_max=None),
    )


def get_criterion(
    feature: np.ndarray,
    candidate: Union[np.ndarray, int],
    ref_point: int,
    threshold: float,
) -> Tuple[Union[np.ndarray, float], float, float]:
    return feature[candidate], feature[ref_point], threshold


def region_growing(
    point_cloud: np.ndarray,
    radius: float,
    omnivariance: np.ndarray,
    planarity: np.ndarray,
    neighborhood_size: np.ndarray,
    moment_x_sq: np.ndarray,
    moment_y_sq: np.ndarray,
    thresholds: Dict[str, float],
    verbose: bool = False,
    use_means: bool = False,
) -> np.ndarray:
    n_points = len(point_cloud)
    region = np.zeros(n_points, dtype=bool)
    visited = np.zeros(n_points, dtype=bool)

    queue = Queue()
    kdtree = KDTree(point_cloud)
    seed = select_seed(point_cloud)

    region[seed] = visited[seed] = True
    queue.put(seed)

    while not queue.empty():
        q = queue.get()
        point = point_cloud[[q]]
        neighbors = kdtree.query_radius(point, radius)[0]

        neighbors = neighbors[np.logical_not(visited[neighbors])]
        visited[neighbors] = True
        if verbose and neighbors.shape[0] > 0:
            print(f"\nNeighborhood size: {neighbors.shape[0]}")

        crit = (
            region_criterion(
                get_avg_criterion(
                    omnivariance, neighbors, region, thresholds["omnivariance"]
                ),
                get_avg_criterion(
                    planarity, neighbors, region, thresholds["planarity"]
                ),
                get_avg_criterion(
                    neighborhood_size,
                    neighbors,
                    region,
                    thresholds["neighborhood_size"],
                ),
                get_avg_criterion(
                    moment_x_sq, neighbors, region, thresholds["moment_x_sq"]
                ),
                get_avg_criterion(
                    moment_y_sq, neighbors, region, thresholds["moment_y_sq"]
                ),
            )
            if use_means
            else region_criterion(
                get_criterion(omnivariance, neighbors, q, thresholds["omnivariance"]),
                get_criterion(planarity, neighbors, q, thresholds["planarity"]),
                get_criterion(
                    neighborhood_size, neighbors, q, thresholds["neighborhood_size"]
                ),
                get_criterion(moment_x_sq, neighbors, q, thresholds["moment_x_sq"]),
                get_criterion(moment_y_sq, neighbors, q, thresholds["moment_y_sq"]),
            )
        )
        if verbose and neighbors.shape[0] > 0:
            print(f"Number of points selected: {crit.sum()} / {neighbors.shape[0]}")

        in_region = neighbors[crit]
        region[in_region] = True

        for selected_neighbor in in_region:
            queue.put(selected_neighbor)

    return region


def aggregate_labels(
    labels: np.ndarray,
    n_labels: int,
    weights: Optional[List[float]] = None,
    in_depth_analysis: bool = False,
) -> int:
    if weights is None:
        weights = [1.0 for _ in range(n_labels)]
        # weights[-3] = 0.1  # let's never predict pedestrian since there are only a few of them
        # weights[1] = 1e-3  # we should not predict the ground since there is no ground anymore

    best_label, best_score = -1, 0
    for label in range(n_labels):
        if (label_score := weights[label] * (labels == label).sum()) > best_score:
            best_label = label
            best_score = label_score

    if in_depth_analysis:
        # minlength is necessary if one label is not represented
        label_scores = np.bincount(labels, minlength=n_labels) * np.array(weights)
        softmax = np.exp(label_scores - np.max(label_scores))
        softmax /= softmax.sum(axis=0)
        print(f"Label weighted proportions: {str(softmax):.2f} ({labels.shape[0]} labels in total).")

        return label_scores.argmax()

    return best_label


@timeit
def smooth_labels(
    cloud: np.ndarray,
    labels: np.ndarray,
    radius: float,
    n_labels: int,
    omnivariance: np.ndarray,
    planarity: np.ndarray,
    neighborhood_size: np.ndarray,
    moment_x_sq: np.ndarray,
    moment_y_sq: np.ndarray,
    thresholds: Optional[Dict[str, float]] = None,
    verbose: bool = False,
) -> None:
    assert cloud.shape[0] == labels.shape[0], "Not the same number of points and labels"
    assert (
        cloud.shape[0] == omnivariance.shape[0]
    ), "Not the same number of points and feature points"

    if thresholds is None:
        thresholds = default_thresholds
    non_smooth_area = np.ones(len(cloud), dtype=bool)
    non_smooth_area[labels == 1] = False

    while non_smooth_area.any():
        if verbose:
            print(f"{non_smooth_area.sum()} points left unvisited.")
        remaining_indices = np.flatnonzero(non_smooth_area)
        region = region_growing(
            cloud[remaining_indices],
            radius,
            omnivariance[remaining_indices],
            planarity[remaining_indices],
            neighborhood_size[remaining_indices],
            moment_x_sq[remaining_indices],
            moment_y_sq[remaining_indices],
            thresholds,
        )
        labels[remaining_indices[region]] = aggregate_labels(
            labels[remaining_indices[region]], n_labels
        )
        non_smooth_area[remaining_indices[region]] = False
        if verbose:
            print(f"{region.sum()} points visited.\n")
