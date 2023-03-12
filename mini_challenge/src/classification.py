import os
from typing import List, Dict

import numpy as np
from dataclasses import dataclass, field

from .descriptors import compute_features
from .perf_monitoring import timeit
from .ply import read_ply
from .subsampling import grid_subsampling


@dataclass
class FeaturesExtractor:
    """
    Class that computes features on point clouds.
    """

    radius: float = 0.1
    n_scales: int = 8
    phi: float = 2
    rho: float = 5
    n_max_points_per_class: int = 50000
    verbose: bool = True
    label_names: Dict[int, str] = field(
        default_factory=lambda: {
            0: "Unclassified",
            1: "Ground",
            2: "Building",
            3: "Poles",
            4: "Pedestrians",
            5: "Cars",
            6: "Vegetation",
        }
    )

    @timeit
    def compute_features(
        self, query_points: np.ndarray, subsampled_clouds: List[np.ndarray]
    ) -> np.ndarray:
        """
        Computes multiscale spherical neighborhoods by applying compute_features on each subsampled point cloud.
        Outputs n_scales times more features than compute_features.
        """
        features = np.empty((len(query_points), 0))
        # computing features for the points of the chosen indices and place them in a [N, 21] matrix
        for scale in range(self.n_scales):
            features = np.hstack(
                (
                    features,
                    compute_features(
                        query_points,
                        subsampled_clouds[scale],
                        self.radius * self.phi**scale,
                    ),
                )
            )

        return features

    def aggregate_features(self, features: np.ndarray) -> np.ndarray:
        """
        Sums an (n_features, n_scales)-array of features over the scales.
        """
        return features.reshape(
            (
                features.shape[0],
                features.shape[1] // self.n_scales,
                self.n_scales,
            )
        ).sum(axis=-1)

    @timeit
    def subsample_point_cloud(self, point_cloud: np.ndarray) -> List[np.ndarray]:
        """
        Sub-samples a point cloud with the radius defined by the multiscale neighborhoods (r_0 * phi ** s).
        """
        subsampled_clouds = [point_cloud]

        for scale in range(1, self.n_scales):
            subsampled_clouds.append(
                grid_subsampling(
                    point_cloud, self.radius * self.phi**scale / self.rho
                )
            )

        return subsampled_clouds

    def sample_indices(self, labels: np.ndarray) -> np.ndarray:
        """
        Samples at most n_max_points_per_class indices for each label to equilibrate the classes.
        """
        indices = np.empty(0, dtype=np.int32)

        # looping over each class to choose training points
        for label, name in self.label_names.items():

            # excluding class 0 in training
            if label == 0:
                continue

            label_indices = np.where(labels == label)[0]
            if self.verbose:
                print(
                    f"{len(label_indices)} elements available for class {self.label_names[label]}"
                )

            # if you do not have enough indices, just take all of them
            if len(label_indices) <= self.n_max_points_per_class:
                indices = np.hstack((indices, label_indices))

            # if you have more than enough indices, choose randomly
            else:
                random_choice = np.random.choice(
                    len(label_indices), self.n_max_points_per_class, replace=False
                )
                indices = np.hstack((indices, label_indices[random_choice]))

        return indices

    def extract_features(
        self, path: str, test_file: str = "", override_cache: bool = False
    ):
        """
        This method extract features/labels of a subset of the training points. It ensures a balanced choice between
        classes.

        Args:
            path: path where the ply files are located.
            test_file: name of the ply file that will be used for validation.
            override_cache: whether cached features should be overriden or not.
        Returns:
            features and labels
        """
        ply_files = [f for f in os.listdir(path) if f.endswith(".ply")]

        train_features = np.empty((0, 21 * self.n_scales))
        train_labels = np.empty((0,))
        test_features = np.empty((0, 21 * self.n_scales))
        test_labels = np.empty((0,))

        for i, file in enumerate(ply_files):
            if self.verbose:
                print(f"\nReading file {file}")

            # caching the features file
            feature_file = os.path.join(path, f"{file[:-4]}_features.npy")
            # the labels have to be cached as well because the indices are picked randomly
            label_file = os.path.join(path, f"{file[:-4]}_labels.npy")

            if (
                os.path.exists(feature_file)
                and os.path.exists(label_file)
                and not override_cache
            ):
                if self.verbose:
                    print("Using cached features and labels")
                features = np.load(feature_file)
                selected_labels = np.load(label_file)
            else:
                cloud_ply = read_ply(os.path.join(path, file))
                points = np.vstack((cloud_ply["x"], cloud_ply["y"], cloud_ply["z"])).T
                labels = cloud_ply["class"]

                training_indices = self.sample_indices(labels)
                selected_labels = labels[training_indices]

                subsampled_clouds = self.subsample_point_cloud(points)
                training_points = points[training_indices, :]
                features = self.compute_features(training_points, subsampled_clouds)

                np.save(feature_file, features)
                np.save(label_file, selected_labels)

            if file == test_file:
                test_features = np.vstack((test_features, features))
                test_labels = np.hstack((test_labels, selected_labels))
            else:
                train_features = np.vstack((train_features, features))
                train_labels = np.hstack((train_labels, selected_labels))

        if test_labels.shape == (0,):
            return train_features, train_labels
        else:
            return train_features, train_labels, test_features, test_labels

    def extract_features_no_label(
        self, path: str, override_cache: bool = False
    ) -> np.ndarray:
        """
        Extracts features of all the test points. Caches the features computed in a npy file.
        """
        ply_files = [f for f in os.listdir(path) if f.endswith(".ply")]

        test_features = np.empty((0, 21 * self.n_scales))

        for i, file in enumerate(ply_files):
            if self.verbose:
                print(f"\nReading file {file}")

            # caching the features file
            feature_file = os.path.join(path, f"{file[:-4]}_features.npy")
            if os.path.exists(feature_file) and not override_cache:
                if self.verbose:
                    print("Using cached features")
                features = np.load(feature_file)
            else:
                cloud_ply = read_ply(os.path.join(path, file))
                points = np.vstack((cloud_ply["x"], cloud_ply["y"], cloud_ply["z"])).T
                subsampled_clouds = self.subsample_point_cloud(points)
                # this part is costly because of the amount of test data
                features = self.compute_features(points, subsampled_clouds)
                np.save(feature_file, features)

            test_features = np.vstack((test_features, features))

        return test_features

    def extract_point_clouds(self, path: str, test_file: str = ""):
        """
        Extracts and split a dataset into test and train.
        The splitting cannot be done arbitrarily when using deep learning, the model could over-fit by learning the
        positions of each group of points of the same label.
        """
        ply_files = [f for f in os.listdir(path) if f.endswith(".ply")]

        train_features = np.empty((0, 3))
        train_labels = np.empty((0,))
        test_features = np.empty((0, 3))
        test_labels = np.empty((0,))

        for i, file in enumerate(ply_files):
            if self.verbose:
                print(f"\nReading file {file}")

            cloud_ply = read_ply(os.path.join(path, file))
            points = np.vstack((cloud_ply["x"], cloud_ply["y"], cloud_ply["z"])).T
            labels = cloud_ply["class"]

            indices = self.sample_indices(labels)

            # one file is designated as the test file
            if file == test_file:
                test_features = np.vstack((test_features, points[indices, :]))
                test_labels = np.hstack((test_labels, labels[indices]))
            else:
                train_features = np.vstack((train_features, points[indices, :]))
                train_labels = np.hstack((train_labels, labels[indices]))

        if test_labels.shape == (0,):
            return train_features, train_labels
        else:
            return train_features, train_labels, test_features, test_labels

    def extract_point_cloud_no_label(self, path: str):
        """
        Extracts the point clouds in the specified folder.
        """
        ply_files = [f for f in os.listdir(path) if f.endswith(".ply")]

        point_cloud = np.empty((0, 3))

        for i, file in enumerate(ply_files):
            if self.verbose:
                print(f"\nReading file {file}")

            cloud_ply = read_ply(os.path.join(path, file))
            points = np.vstack((cloud_ply["x"], cloud_ply["y"], cloud_ply["z"])).T
            point_cloud = np.vstack((point_cloud, points))

        return point_cloud
