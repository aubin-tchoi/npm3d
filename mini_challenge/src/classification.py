import os
import time
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .descriptors import compute_features
from .ply import read_ply


class FeaturesExtractor:
    """
    Class that computes features from point clouds
    """

    def __init__(self):
        """
        Initiation method called when an object of this class is created. This is where you can define parameters
        """

        # Neighborhood radius
        self.radius = 0.5

        # Number of training points per class
        self.num_per_class = 500

        # Classification labels
        self.label_names = {
            0: "Unclassified",
            1: "Ground",
            2: "Building",
            3: "Poles",
            4: "Pedestrians",
            5: "Cars",
            6: "Vegetation",
        }

    def extract_training(self, path):
        """
        This method extract features/labels of a subset of the training points. It ensures a balanced choice between
        classes.
        Args:
            path: path where the ply files are located.
        Returns:
            features and labels
        """

        # Get all the ply files in data folder
        ply_files = [f for f in os.listdir(path) if f.endswith(".ply")]

        # Initiate arrays
        training_features = np.empty((0, 21))
        training_labels = np.empty((0,))

        # Loop over each training cloud
        for i, file in enumerate(ply_files):

            # Load Training cloud
            cloud_ply = read_ply(os.path.join(path, file))
            points = np.vstack((cloud_ply["x"], cloud_ply["y"], cloud_ply["z"])).T
            labels = cloud_ply["class"]

            # Initiate training indices array
            training_inds = np.empty(0, dtype=np.int32)

            # Loop over each class to choose training points
            for label, name in self.label_names.items():

                # Do not include class 0 in training
                if label == 0:
                    continue

                # Collect all indices of the current class
                label_inds = np.where(labels == label)[0]

                # If you have not enough indices, just take all of them
                if len(label_inds) <= self.num_per_class:
                    training_inds = np.hstack((training_inds, label_inds))

                # If you have more than enough indices, choose randomly
                else:
                    random_choice = np.random.choice(
                        len(label_inds), self.num_per_class, replace=False
                    )
                    training_inds = np.hstack(
                        (training_inds, label_inds[random_choice])
                    )

            # Gather chosen points
            training_points = points[training_inds, :]

            # Compute features for the points of the chosen indices and place them in a [N, 21] matrix
            features = compute_features(training_points, points, self.radius)

            # Concatenate features / labels of all clouds
            training_features = np.vstack((training_features, features))
            training_labels = np.hstack((training_labels, labels[training_inds]))

        return training_features, training_labels

    def extract_data(self, path: str, test_file: str):
        ply_files = [f for f in os.listdir(path) if f.endswith(".ply")]

        train_features = np.empty((0, 3))
        train_labels = np.empty((0,))
        test_features = np.empty((0, 3))
        test_labels = np.empty((0,))

        # Loop over each training cloud
        for i, file in enumerate(ply_files):
            print(f"Reading file {file}")

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

                # if you do not have enough indices, just take all of them
                if len(label_inds) <= self.num_per_class:
                    indices = np.hstack((indices, label_inds))

                # if you have more than enough indices, choose randomly
                else:
                    random_choice = np.random.choice(
                        len(label_inds), self.num_per_class, replace=False
                    )
                    indices = np.hstack((indices, label_inds[random_choice]))

            if file == test_file:
                test_features = np.vstack((test_features, points[indices, :]))
                test_labels = np.hstack((test_labels, labels[indices]))
            else:
                train_features = np.vstack((train_features, points[indices, :]))
                train_labels = np.hstack((train_labels, labels[indices]))

        return train_features, train_labels, test_features, test_labels

    @staticmethod
    def extract_data_no_label(path):
        # Get all the ply files in data folder
        ply_files = [f for f in os.listdir(path) if f.endswith(".ply")]

        # Initiate arrays
        features = np.empty((0, 3))

        # Loop over each training cloud
        for i, file in enumerate(ply_files):
            cloud_ply = read_ply(os.path.join(path, file))
            points = np.vstack((cloud_ply["x"], cloud_ply["y"], cloud_ply["z"])).T
            features = np.vstack((features, points))

        return features

    def extract_test(self, path):
        """
        This method extract features of all the test points.
        :param path: path where the ply files are located.
        :return: features
        """

        # Get all the ply files in data folder
        ply_files = [f for f in os.listdir(path) if f.endswith(".ply")]

        # Initiate arrays
        test_features = np.empty((0, 21))

        # Loop over each training cloud
        for i, file in enumerate(ply_files):

            # Load Training cloud
            cloud_ply = read_ply(os.path.join(path, file))
            points = np.vstack((cloud_ply["x"], cloud_ply["y"], cloud_ply["z"])).T

            # Compute features only one time and save them for further use
            #
            #   WARNING : This will save you some time but do not forget to delete your features file if you change
            #   your features. Otherwise, you will not compute them and use the previous ones
            #

            # Name the feature file after the ply file.
            feature_file = file[:-4] + "_features.npy"
            feature_file = os.path.join(path, feature_file)

            # If the file exists load the previously computed features
            if os.path.exists(os.path.join(path, feature_file)):
                features = np.load(feature_file)

            # If the file does not exist, compute the features (very long) and save them for future use
            else:
                features = compute_features(points, points, self.radius)
                np.save(feature_file, features)

            # Concatenate features of several clouds
            # (For this minichallenge this is useless as the test set contains only one cloud)
            test_features = np.vstack((test_features, features))

        return test_features


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
    training_features, training_labels = f_extractor.extract_training(training_path)
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
    test_features = f_extractor.extract_test(test_path)
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
