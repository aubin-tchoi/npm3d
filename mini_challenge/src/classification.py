import os
import time
from datetime import datetime
from typing import List

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .descriptors import compute_features
from .ply import read_ply
from .subsampling import grid_subsampling
from .perf_monitoring import timeit


class FeaturesExtractor:
    """
    Class that computes features from point clouds.
    """

    def __init__(self, verbose: bool = True):
        """
        Initiation method called when an object of this class is created. This is where you can define parameters
        """

        # subsampling and neighborhood parameters
        self.radius = 0.1
        self.n_scales = 8
        self.phi = 2
        self.rho = 5

        # number of training points per class
        self.num_per_class = 50000

        # classification labels
        self.label_names = {
            0: "Unclassified",
            1: "Ground",
            2: "Building",
            3: "Poles",
            4: "Pedestrians",
            5: "Cars",
            6: "Vegetation",
        }

        self.verbose = verbose

    @timeit
    def compute_features(
        self, query_points: np.ndarray, subsampled_clouds: List[np.ndarray]
    ) -> np.ndarray:

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

    @timeit
    def subsample_point_cloud(self, point_cloud: np.ndarray) -> List[np.ndarray]:
        subsampled_clouds = [point_cloud]

        for scale in range(1, self.n_scales):
            subsampled_clouds.append(
                grid_subsampling(
                    point_cloud, self.radius * self.phi**scale / self.rho
                )
            )

        return subsampled_clouds

    def extract_features(self, path, test_file: str = "", override_cache: bool = False):
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

            cloud_ply = read_ply(os.path.join(path, file))
            points = np.vstack((cloud_ply["x"], cloud_ply["y"], cloud_ply["z"])).T
            labels = cloud_ply["class"]

            training_inds = np.empty(0, dtype=np.int32)

            for label, name in self.label_names.items():

                # class 0 is excluded from training
                if label == 0:
                    continue

                label_inds = np.where(labels == label)[0]
                if self.verbose:
                    print(
                        f"{len(label_inds)} elements available for class {self.label_names[label]}"
                    )

                # taking all the indices if there is not enough of them
                if len(label_inds) <= self.num_per_class:
                    training_inds = np.hstack((training_inds, label_inds))

                # choosing randomly otherwise
                else:
                    random_choice = np.random.choice(
                        len(label_inds), self.num_per_class, replace=False
                    )
                    training_inds = np.hstack(
                        (training_inds, label_inds[random_choice])
                    )

            # caching the features file
            feature_file = os.path.join(path, f"{file[:-4]}_features.npy")
            if os.path.exists(os.path.join(path, feature_file)) and not override_cache:
                features = np.load(feature_file)
            else:
                subsampled_clouds = self.subsample_point_cloud(points)
                training_points = points[training_inds, :]
                features = self.compute_features(training_points, subsampled_clouds)
                np.save(feature_file, features)

            if file == test_file:
                test_features = np.vstack((test_features, features))
                test_labels = np.hstack((test_labels, labels[training_inds]))
            else:
                train_features = np.vstack((train_features, features))
                train_labels = np.hstack((train_labels, labels[training_inds]))

        return train_features, train_labels, test_features, test_labels

    def extract_features_no_label(self, path: str, override_cache: bool = False):
        """
        Extracts features of all the test points. Caches the features computed in a npy file.
        """

        ply_files = [f for f in os.listdir(path) if f.endswith(".ply")]

        test_features = np.empty((0, 21 * self.n_scales))

        for i, file in enumerate(ply_files):
            if self.verbose:
                print(f"\nReading file {file}")

            cloud_ply = read_ply(os.path.join(path, file))
            points = np.vstack((cloud_ply["x"], cloud_ply["y"], cloud_ply["z"])).T

            # caching the features file
            feature_file = os.path.join(path, f"{file[:-4]}_features.npy")
            if os.path.exists(os.path.join(path, feature_file)) and not override_cache:
                features = np.load(feature_file)
            else:
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

            indices = np.empty(0, dtype=np.int32)

            # looping over each class to choose training points
            for label, name in self.label_names.items():

                # excluding class 0 in training
                if label == 0:
                    continue

                label_inds = np.where(labels == label)[0]
                if self.verbose:
                    print(
                        f"{len(label_inds)} elements available for class {self.label_names[label]}"
                    )

                # if you do not have enough indices, just take all of them
                if len(label_inds) <= self.num_per_class:
                    indices = np.hstack((indices, label_inds))

                # if you have more than enough indices, choose randomly
                else:
                    random_choice = np.random.choice(
                        len(label_inds), self.num_per_class, replace=False
                    )
                    indices = np.hstack((indices, label_inds[random_choice]))

            # one file is designated as the test file
            if file == test_file:
                test_features = np.vstack((test_features, points[indices, :]))
                test_labels = np.hstack((test_labels, labels[indices]))
            else:
                train_features = np.vstack((train_features, points[indices, :]))
                train_labels = np.hstack((train_labels, labels[indices]))

        return train_features, train_labels, test_features, test_labels

    def extract_point_cloud_no_label(self, path: str):
        ply_files = [f for f in os.listdir(path) if f.endswith(".ply")]

        point_cloud = np.empty((0, 3))

        for i, file in enumerate(ply_files):
            if self.verbose:
                print(f"\nReading file {file}")

            cloud_ply = read_ply(os.path.join(path, file))
            points = np.vstack((cloud_ply["x"], cloud_ply["y"], cloud_ply["z"])).T
            point_cloud = np.vstack((point_cloud, points))

        return point_cloud


if __name__ == "__main__":
    # paths of the training and test files
    training_path = "../data/training"
    test_path = "../data/test"

    #   For this simple algorithm, we only compute the features for a subset of the training points. We choose N points
    #   per class in each training file. This has two advantages : balancing the class for our classifier and saving a
    #   lot of computational time.

    print("Collect Training Features")
    t0 = time.time()
    f_extractor = FeaturesExtractor()
    training_features, training_labels, _, __ = f_extractor.extract_features(
        training_path
    )
    t1 = time.time()
    print("Done in %.3fs\n" % (t1 - t0))

    print("Training Random Forest")
    t0 = time.time()
    clf = RandomForestClassifier()
    clf.fit(training_features, training_labels)
    t1 = time.time()
    print("Done in %.3fs\n" % (t1 - t0))

    print("Compute testing features")
    t0 = time.time()
    test_features = f_extractor.extract_features_no_label(test_path)
    t1 = time.time()
    print("Done in %.3fs\n" % (t1 - t0))

    print("Test")
    t0 = time.time()
    predictions = clf.predict(test_features)
    t1 = time.time()
    print("Done in %.3fs\n" % (t1 - t0))

    assert predictions.shape[0] == 3079187, "Incorrect number of predictions"

    print("Save predictions")
    t0 = time.time()
    np.savetxt(
        f"submissions/feat-{datetime.now().strftime('%Y_%m_%d-%H_%M')}.txt",
        predictions,
        fmt="%d",
    )
    t1 = time.time()
    print("Done in %.3fs\n" % (t1 - t0))
