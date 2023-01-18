"""
Basic transformation on a point cloud.
"""
import numpy as np


from ply import write_ply, read_ply


if __name__ == "__main__":

    # loading the point cloud
    file_path = "../data/bunny.ply"
    data = read_ply(file_path)

    # concatenating x, y, and z in a (N*3) point matrix
    points = np.vstack((data["x"], data["y"], data["z"])).T

    # concatenating R, G, and B channels in a (N*3) color matrix
    colors = np.vstack((data["red"], data["green"], data["blue"])).T

    # transforming the point cloud
    centroid = np.mean(points, axis=0)
    points -= centroid
    # we can directly divide by 2 since the points are centered
    points /= 2
    points += centroid
    points -= np.array([0, 0.1, 0])

    transformed_points = points

    # saving the point cloud
    write_ply(
        "../little_bunny.ply",
        [transformed_points, colors],
        ["x", "y", "z", "red", "green", "blue"],
    )

    print("Done")
