import numpy as np
from sklearn.neighbors import KDTree

# setting a seed
rng = np.random.default_rng(seed=1)


def random_sample(points: np.ndarray) -> np.ndarray:
    """
    Randomly samples three points using indices.
    """
    return points[np.random.choice(len(points), 3, replace=False)]


def fast_random_sample(points: np.ndarray) -> np.ndarray:
    """
    Randomly samples three points using a random Generator.
    This method is significantly faster than the previous one (approximately two times faster on most cases).
    """
    return rng.choice(points, 3, replace=False, shuffle=False)


def local_sample(points: np.ndarray, kdtree: KDTree, k: int = 5000) -> np.ndarray:
    """
    Randomly samples three points in the vicinity of one another using indices.
    """
    first_point = points[np.random.choice(len(points))].reshape(1, 3)
    neighborhood = points[
        kdtree.query(first_point, k=k, return_distance=False).squeeze()[
            np.random.choice(k, 2, replace=False)
        ]
    ]

    return np.vstack((first_point, neighborhood))


def fast_local_sample(points: np.ndarray, kdtree: KDTree, k: int = 5000) -> np.ndarray:
    """
    Randomly samples three points in the vicinity of one another using a random Generator.
    This method is not significantly faster than the previous one, but it enables seed selection.
    """
    first_point = rng.choice(points).reshape(1, 3)
    neighborhood = points[
        kdtree.query(first_point, k=k, return_distance=False).squeeze()[
            rng.choice(k, 2, replace=False)
        ]
    ]

    return np.vstack((first_point, neighborhood))
