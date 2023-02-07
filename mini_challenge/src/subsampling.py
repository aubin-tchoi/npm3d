import numpy as np


def grid_subsampling(
    points: np.ndarray, voxel_size: float, verbose: bool = False
) -> np.ndarray:
    """
    Performs a voxel subsampling on the point cloud. Keeps the barycenter of the points in each voxel.
    """
    nb_vox = np.ceil((np.max(points - np.min(points, axis=0), axis=0)) / voxel_size)
    if verbose:
        print(
            f"Number of voxels: {'; '.join(nb_vox.astype(int).astype(str))} (total: {int(nb_vox.prod()):,})".replace(
                ",", " "
            ).replace(
                ";", ","
            )
        )
    non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(
        ((points - np.min(points, axis=0)) // voxel_size).astype(int),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    if verbose:
        print(
            f"Number of nonempty voxels: {len(non_empty_voxel_keys):,}".replace(
                ",", " "
            )
        )

    idx_pts_vox_sorted = np.argsort(inverse)

    sub_sampled_points = np.zeros((len(non_empty_voxel_keys), 3), points.dtype)

    last_seen = 0
    for idx in range(len(non_empty_voxel_keys)):
        indexes_in_voxel = idx_pts_vox_sorted[
            last_seen : last_seen + nb_pts_per_voxel[idx]
        ]
        # barycenter of points
        sub_sampled_points[idx] = points[indexes_in_voxel].mean(axis=0)

        last_seen += nb_pts_per_voxel[idx]

    return sub_sampled_points
