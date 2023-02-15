"""
Implementation of local 3D point cloud surface reconstruction algorithms.
"""

import argparse
import time

import numpy as np
import trimesh
from skimage import measure
from sklearn.neighbors import KDTree

from ply import read_ply


def compute_hoppe(points, normals, scalar_field, grid_resolution, min_grid, size_voxel):
    my_tree = KDTree(points, 10)

    d = points.shape[1]  # should be equal to 3
    coords = np.arange(grid_resolution)
    voxels = np.stack(np.meshgrid(*d * [coords], indexing="ij"), axis=-1)
    voxels = (voxels * size_voxel + min_grid).astype(np.float32).reshape(-1, d)

    indices = my_tree.query(voxels, return_distance=False).squeeze()

    # Hoppe function
    volume = np.sum(normals[indices] * (voxels - points[indices]), axis=-1)

    scalar_field[:, :, :] = volume.reshape(*d * [grid_resolution])


def compute_imls(
    points, normals, scalar_field, grid_resolution, min_grid, size_voxel, knn, h=0.01
):
    my_tree = KDTree(points, 10)
    d = points.shape[1]  # should be equal to 3

    coords = np.arange(grid_resolution)
    voxels = np.stack(np.meshgrid(*d * [coords], indexing="ij"), axis=-1)
    voxels = (voxels * size_voxel + min_grid).astype(np.float32).reshape(-1, d)

    distances, indices = my_tree.query(voxels, k=knn)

    # computing theta once for all because it is used twice
    theta = np.exp(-((distances / h) ** 2))
    hoppe = np.sum(
        normals[indices] * (voxels[:, np.newaxis] - points[indices]), axis=-1
    )
    volume = np.sum(hoppe * theta, axis=-1) / (np.sum(theta, axis=-1) + 1e-16)

    scalar_field[:, :, :] = volume.reshape(*d * [grid_resolution])


def parse_args() -> argparse.Namespace:
    """
    Parses the command line arguments. Also produces the help message.
    """
    parser = argparse.ArgumentParser(description="Surface reconstruction")

    parser.add_argument(
        "--grid_resolution",
        type=int,
        default=128,
        help="Grid resolution",
    )
    parser.add_argument(
        "--implicit_function",
        type=str,
        default="imls",
        help="Implicit function to use in the reconstruction",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    t0 = time.time()

    file_path = "../data/bunny_normals.ply"
    data = read_ply(file_path)
    points = np.vstack((data["x"], data["y"], data["z"])).T
    normals = np.vstack((data["nx"], data["ny"], data["nz"])).T

    min_grid = np.amin(points, axis=0)
    max_grid = np.amax(points, axis=0)

    # increasing the bounding box of data points by decreasing min_grid and inscreasing max_grid
    min_grid = min_grid - 0.10 * (max_grid - min_grid)
    max_grid = max_grid + 0.10 * (max_grid - min_grid)

    # grid_resolution is the number of voxels in the grid in x, y, z axis
    grid_resolution = args.grid_resolution  # 128
    size_voxel = np.array(
        [
            (max_grid[0] - min_grid[0]) / (grid_resolution - 1),
            (max_grid[1] - min_grid[1]) / (grid_resolution - 1),
            (max_grid[2] - min_grid[2]) / (grid_resolution - 1),
        ]
    )

    # creating a volume grid to compute the scalar field for surface reconstruction
    scalar_field = np.zeros(
        (grid_resolution, grid_resolution, grid_resolution), dtype=np.float32
    )

    # computing the scalar field in the grid
    if args.implicit_function.lower() == "hoppe":
        compute_hoppe(
            points, normals, scalar_field, grid_resolution, min_grid, size_voxel
        )
    elif args.implicit_function.lower() == "imls":
        compute_imls(
            points, normals, scalar_field, grid_resolution, min_grid, size_voxel, 30
        )
    else:
        raise ValueError("Incorrect implicit function name.")

    # computing the mesh from the scalar field using the marching cubes algorithm
    verts, faces, normals_tri, values_tri = measure.marching_cubes(
        scalar_field, level=0.0, spacing=(size_voxel[0], size_voxel[1], size_voxel[2])
    )

    print(f"Total number of faces: {len(faces)}.")

    # exporting the mesh in ply using trimesh
    mesh = trimesh.Trimesh(vertices=verts + min_grid, faces=faces)
    mesh.export(
        file_obj=f"../bunny_mesh_{args.implicit_function}_{grid_resolution}.ply",
        file_type="ply",
    )

    print("Total time for surface reconstruction : ", time.time() - t0)
