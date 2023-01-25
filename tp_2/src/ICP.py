"""
Implementation of the ICP method.
"""
from time import perf_counter
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree

from ply import write_ply, read_ply
from utils import parse_args
from visu import show_ICP


def compare_H_computation_methods(data, ref):
    """
    Compares three methods to compute H:
        - use np.einsum to perform the sum of the outer products of the columns and sum over the rows.
        - do the same using a sum and np.outer.
        - performing a direct matrix multiplication.
    In my experiments, the second method is always significantly slower than the other two, and the matrix
    multiplication always outperforms the einsum except in the very first case of the bunny that was flipped over.
    """
    n_runs = 10
    method_times = [0, 0, 0]

    for i in range(n_runs):
        t_ref = perf_counter()
        einsum_res = np.einsum(
            "ji,ki->jk",
            data,
            ref,
        )
        method_times[0] += perf_counter() - t_ref

        t_ref = perf_counter()
        sum_outer_res = sum(
            np.outer(
                data[:, i],
                ref[:, i],
            )
            for i in range(data.shape[1])
        )
        method_times[1] += perf_counter() - t_ref

        t_ref = perf_counter()
        matmul_res = data.squeeze() @ ref.T
        method_times[2] += perf_counter() - t_ref

        assert np.allclose(
            einsum_res, sum_outer_res
        ), "H matrix returned is not the same between two methods"
        assert np.allclose(
            sum_outer_res, matmul_res
        ), "H matrix returned is not the same between two methods"

    print("Comparison between the three methods:")
    method_names = ["np.einsum", "sum + np.outer", "matmul"]
    for i in range(len(method_times)):
        print(f"{method_names[i]:>14}: {method_times[i] / n_runs * 1000:.4f} ms")
    print(f"-- Best: {method_names[np.argmin(method_times)]}\n")


def best_rigid_transform(data, ref):
    """
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
         ref = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    """

    data_barycenter = data.mean(axis=1)
    ref_barycenter = ref.mean(axis=1)
    H = (data - data_barycenter[:, np.newaxis]).squeeze() @ (
        ref - ref_barycenter[:, np.newaxis]
    ).squeeze().T
    U, S, V = np.linalg.svd(H)
    R = V.T @ U.T

    # ensuring that we have a direct rotation (determinant equal to 1 and not -1)
    if np.linalg.det(R) < 0:
        U_transpose = U.T
        U_transpose[-1] *= -1
        R = V.T @ U_transpose

    T = ref_barycenter - R.dot(data_barycenter)

    return R, T


def icp_point_to_point(data, ref, max_iter, RMS_threshold):
    """
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration

    """
    data_aligned = np.copy(data)
    kdtree = KDTree(ref.T)

    R_list, T_list = [], []
    neighbors_list, RMS_list = [], []

    for i in range(max_iter):
        neighbors = kdtree.query(data_aligned.T, return_distance=False).squeeze()
        R, T = best_rigid_transform(
            data_aligned,
            ref[:, neighbors],
        )
        data_aligned = R.dot(data_aligned) + T[:, np.newaxis]
        rms = np.sqrt(np.sum((data_aligned - ref[:, neighbors]) ** 2, axis=0).mean())

        R_list.append(R)
        T_list.append(T[:, np.newaxis])
        neighbors_list.append(neighbors)
        RMS_list.append(rms)

        if rms < RMS_threshold:
            print("RMS threshold reached.")
            break
    # for ... else clause executed if we did not break out of the loop
    else:
        print("Max iteration number reached.")

    return data_aligned, R_list, T_list, neighbors_list, RMS_list


def icp_point_to_point_fast(
    data: np.ndarray,
    ref: np.ndarray,
    max_iter: int,
    RMS_threshold: float,
    sampling_limit: int,
):
    """
    Iterative closest point algorithm with a point to point strategy.
    Each iteration is performed on a subsampled of the point clouds to fasten the computation.

    Args:
        data: (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref: (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter: stop condition on the number of iterations
        RMS_threshold: stop condition on the distance
        sampling_limit: number of points used at each iteration.

    Returns:
        data_aligned: data aligned on reference cloud
        R_list: list of the (d x d) rotation matrices found at each iteration
        T_list: list of the (d x 1) translation vectors found at each iteration
        neighbors_list: At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
    """
    data_aligned = np.copy(data)
    kdtree = KDTree(ref.T)

    R_list, T_list = [], []
    neighbors_list, RMS_list = [], []
    rms = 0.0

    for i in range(max_iter):
        indexes = np.random.choice(data.shape[1], sampling_limit, replace=False)
        data_aligned_subset = data_aligned[:, indexes]
        neighbors = kdtree.query(data_aligned_subset.T, return_distance=False).squeeze()
        R, T = best_rigid_transform(
            data_aligned_subset,
            ref[:, neighbors],
        )
        data_aligned = R.dot(data_aligned) + T[:, np.newaxis]
        rms = np.sqrt(
            np.sum(
                (data_aligned_subset - ref[:, neighbors]) ** 2,
                axis=0,
            ).mean()
        )

        R_list.append(R)
        T_list.append(T[:, np.newaxis])
        neighbors_list.append(neighbors)
        RMS_list.append(rms)

        if rms < RMS_threshold:
            print("RMS threshold reached.")
            break
    # for ... else clause executed if we did not break out of the loop
    else:
        print("Max iteration number reached.")

    print(f"Final RMS: {rms:.4f}")

    return data_aligned, R_list, T_list, neighbors_list, RMS_list


def get_data(data_path: str, ref_path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = read_ply(data_path)
    ref = read_ply(ref_path)

    return (
        np.vstack((data["x"], data["y"], data["z"])),
        np.vstack((ref["x"], ref["y"], ref["z"])),
    )


if __name__ == "__main__":
    args = parse_args()

    if not args.skip_rigid:
        bunny_o_path = "../data/bunny_original.ply"
        bunny_r_path = "../data/bunny_returned.ply"

        # loading the point clouds
        bunny_o, bunny_r = get_data(bunny_o_path, bunny_r_path)

        # finding the best transformation
        R, T = best_rigid_transform(bunny_r, bunny_o)

        # applying the transformation
        bunny_r_opt = R.dot(bunny_r) + T[:, np.newaxis]

        # saving the cloud
        write_ply("../bunny_r_opt", [bunny_r_opt.T], ["x", "y", "z"])

        # computing the RMS
        distances2_before = np.sum(np.power(bunny_r - bunny_o, 2), axis=0)
        RMS_before = np.sqrt(np.mean(distances2_before))
        distances2_after = np.sum(np.power(bunny_r_opt - bunny_o, 2), axis=0)
        RMS_after = np.sqrt(np.mean(distances2_after))

        print("Average RMS between points :")
        print("Before = {:.3f}".format(RMS_before))
        print(" After = {:.3f}".format(RMS_after))

    if not args.skip_2d:
        ref2D_path = "../data/ref2D.ply"
        data2D_path = "../data/data2D.ply"

        # loading the point clouds
        ref2D, data2D = get_data(ref2D_path, data2D_path)

        # applying the ICP
        data2D_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(
            data2D, ref2D, args.max_iter, args.rms_threshold
        )

        # saving the point cloud
        write_ply("../data2D_opt", [data2D_opt.T], ["x", "y"])

        # showing the ICP
        show_ICP(data2D, ref2D, R_list, T_list, neighbors_list)

        # plotting the RMS
        plt.plot(RMS_list)
        plt.title("RMS during ICP on 2D example")
        plt.xlabel("iterations")
        plt.ylabel("RMS")
        plt.show()

    if not args.skip_bunnies:
        bunny_o_path = "../data/bunny_original.ply"
        bunny_p_path = "../data/bunny_perturbed.ply"

        bunny_o, bunny_p = get_data(bunny_o_path, bunny_p_path)

        bunny_p_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(
            bunny_p, bunny_o, args.max_iter, args.rms_threshold
        )

        show_ICP(bunny_p, bunny_o, R_list, T_list, neighbors_list)

        plt.plot(RMS_list)
        plt.title("RMS during ICP between the bunnies")
        plt.xlabel("iterations")
        plt.ylabel("RMS")
        plt.show()

    if not args.skip_nddc:
        notre_dame_1_path = "../data/Notre_Dame_Des_Champs_1.ply"
        notre_dame_2_path = "../data/Notre_Dame_Des_Champs_2.ply"

        notre_dame_1, notre_dame_2 = get_data(notre_dame_1_path, notre_dame_2_path)

        for sampling_limit in [1000, 10000, 50000]:
            (notre_dame_opt, _, __, ___, RMS_list,) = icp_point_to_point_fast(
                notre_dame_1,
                notre_dame_2,
                args.max_iter,
                args.rms_threshold,
                sampling_limit=sampling_limit,
            )

            plt.plot(RMS_list, label=f"sampling limit: {sampling_limit}")

        plt.title("RMS during ICP on NDDC with various sampling limits")
        plt.xlabel("iterations")
        plt.ylabel("RMS")
        plt.legend()
        plt.show()
