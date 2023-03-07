from queue import Queue
from typing import Union, List, Optional

import numpy as np
from sklearn.neighbors import KDTree

from .perf_monitoring import timeit


def region_criterion(
    p1, p2, n1, n2, threshold_dist: float = 0.1, threshold_angle: float = 0.1
):
    norm1 = np.maximum(np.linalg.norm(n1, axis=-1), 1e-10)[:, np.newaxis]
    norm2 = np.maximum(np.linalg.norm(n2, axis=-1), 1e-10)[:, np.newaxis]

    distance = np.abs((p1 - p2) @ n2.T) / norm2.T
    angle = np.arccos(np.clip((n1 @ n2.T) / (norm1 @ norm2.T), -1, 1))

    return (distance < threshold_dist) * (np.abs(angle) < threshold_angle)


def select_seed(point_cloud: np.ndarray) -> Union[np.ndarray, int]:
    return 0


def region_growing(point_cloud: np.ndarray, radius: float):
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
        region[neighbors] = True

        for selected_neighbor in neighbors:
            queue.put(selected_neighbor)

    return region


def aggregate_labels(
    labels: np.ndarray, n_labels: int, weights: Optional[List[float]] = None
) -> int:
    if weights is None:
        weights = [1.0 for _ in range(n_labels)]

    best_label, best_score = -1, 0
    for label in range(n_labels):
        if (label_score := weights[label] * (labels == label).sum()) > best_score:
            best_label = label
            best_score = label_score

    return best_label


@timeit
def smooth_labels(
    cloud: np.ndarray,
    labels: np.ndarray,
    radius: float,
    n_labels: int,
) -> None:
    non_smooth_area = np.ones(len(cloud), dtype=bool)
    non_smooth_area[labels == 1] = False

    while non_smooth_area.any():
        print(f"{non_smooth_area.sum()} points unvisited.")
        remaining_indices = np.flatnonzero(non_smooth_area)
        region = region_growing(
            cloud[remaining_indices],
            radius,
        )
        labels[remaining_indices[region]] = aggregate_labels(
            labels[remaining_indices[region]], n_labels
        )
        non_smooth_area[remaining_indices[region]] = False
        print(f"{region.sum()} points visited.\n")
