from queue import Queue
from typing import Union
import numpy as np
from sklearn.neighbors import KDTree


def region_criterion(
    p1, p2, n1, n2, threshold_dist: float = 0.1, threshold_angle: float = 0.1
):
    norm1 = np.maximum(np.linalg.norm(n1, axis=-1), 1e-10)[:, np.newaxis]
    norm2 = np.maximum(np.linalg.norm(n2, axis=-1), 1e-10)[:, np.newaxis]

    distance = np.abs((p1 - p2) @ n2.T) / norm2.T
    angle = np.arccos(np.clip((n1 @ n2.T) / (norm1 @ norm2.T), -1, 1))

    return (distance < threshold_dist) * (np.abs(angle) < threshold_angle)


def select_seed(point_cloud: np.ndarray) -> Union[np.ndarray, int]:
    pass


def region_growing(point_cloud, normals, radius: float):
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
        point, normal = point_cloud[[q]], normals[[q]]
        neighbors = kdtree.query_radius(point, radius)[0]

        neighbors = neighbors[np.logical_not(visited[neighbors])]
        visited[neighbors] = True

        in_region = neighbors[
            region_criterion(point_cloud[neighbors], point, normals[neighbors], normal)
        ]
        region[in_region] = True

        for selected_neighbor in in_region:
            queue.put(selected_neighbor)

    return region


def multi_region_growing(cloud, normals, radius, n_seeds: int):
    labels = -np.ones(len(cloud), dtype=np.int32)

    for seed in range(n_seeds):
        remaining_indices = np.flatnonzero(labels == -1)

        region = region_growing(
            cloud[remaining_indices],
            normals[remaining_indices],
            radius,
        )
        labels[remaining_indices[region]] = seed

    return labels
