import argparse
import warnings
from datetime import datetime

from lightgbm import LGBMClassifier
from numpy import savetxt

from src import (
    FeaturesExtractor,
    smooth_labels,
    checkpoint,
)

warnings.simplefilter(action="ignore", category=FutureWarning)


def parse_args() -> argparse.Namespace:
    """
    Parses the command line arguments. Also produces the help message.
    """
    parser = argparse.ArgumentParser(
        description="Launches runs of semantic classification of 3D point clouds."
    )

    # I/O options
    parser.add_argument(
        "--identifier",
        type=str,
        default="my_model",
        help="Identifier that will be added to the submission file's name",
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="./data/training",
        help="Path to the folder that contains the train data as ply files",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="./data/test",
        help="Path to the folder that contains the test data as ply files",
    )
    parser.add_argument(
        "--disable_predictions_save",
        action="store_true",
        help="Skips writing the results in a txt file",
    )
    parser.add_argument(
        "--no_verbose",
        action="store_true",
        help="Less verbosity (execution times will still be printed)",
    )

    # parameters for the multiscale neighborhoods
    parser.add_argument(
        "--multiscale_radius",
        type=float,
        default=0.1,
        help="Initial neighborhood radius in the multiscale neighborhoods part",
    )
    parser.add_argument(
        "--n_scales",
        type=int,
        default=8,
        help="Number of scales in the multiscale neighborhoods part",
    )
    parser.add_argument(
        "--phi",
        type=float,
        default=2,
        help="Scaling parameter in the multiscale neighborhoods part",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=5,
        help="Subsampling parameter in the multiscale neighborhoods part",
    )

    # learning parameters
    parser.add_argument(
        "--n_max_points_per_class",
        type=int,
        default=50000,
        help="Maximal number of points chosen in each class",
    )
    parser.add_argument(
        "--n_leaves",
        type=int,
        default=167,
        help="Number of leaves in the LGBM classifier",
    )
    parser.add_argument(
        "--reg_lambda",
        type=float,
        default=0.1,
        help="L2 regularization strength in the LGBM classifier",
    )

    # parameters for the smoothing algorithm
    parser.add_argument(
        "--disable_smoothing",
        action="store_true",
        help="Disables the smoothing part of the method",
    )
    parser.add_argument(
        "--smoothing_radius",
        type=float,
        default=0.2,
        help="Neighborhood radius in the region growing algorithm of the smoothing part",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    features_extractor = FeaturesExtractor(
        radius=args.multiscale_radius,
        n_scales=args.n_scales,
        phi=args.phi,
        rho=args.rho,
        n_max_points_per_class=args.n_max_points_per_class,
        verbose=not args.no_verbose,
    )
    # additional parameters can be added here
    model = LGBMClassifier(num_leaves=args.n_leaves, reg_lambda=args.reg_lambda)

    timer = checkpoint()

    train_features, train_labels = features_extractor.extract_features(
        args.train_data_path
    )
    timer("Time spent computing train features")

    model.fit(train_features, train_labels)
    timer("Time spent fitting the model")

    test_features = features_extractor.extract_features_no_label(args.test_data_path)
    timer("Time spent computing test features")

    predicted_labels = model.predict(test_features)
    timer("Time spent on prediction on the test data")

    aggregated_test_features = features_extractor.aggregate_features(test_features)
    timer("Time spent aggregating the features over the scales")

    point_cloud = features_extractor.extract_point_cloud_no_label(args.test_data_path)
    timer("Time spent extracting the test point cloud")

    if not args.disable_smoothing:
        smooth_labels(
            point_cloud,
            predicted_labels,
            args.smoothing_radius,
            len(features_extractor.label_names),
            aggregated_test_features[:, 2],
            aggregated_test_features[:, 5],
            aggregated_test_features[:, 13],
            aggregated_test_features[:, 15],
            aggregated_test_features[:, 20],
            verbose=not args.no_verbose,
            # the thresholds below can be further fine-tuned
            thresholds={
                "omnivariance": 30,
                "planarity": 75,
                "neighborhood_size": 300,
                "moment_x_sq": 500,
                "moment_y_sq": 500,
            },
        )
        timer("Time spent smoothing the features")

    assert predicted_labels.shape[0] == 3079187, "Incorrect number of predictions"

    if not args.disable_predictions_save:
        savetxt(
            (
                file_path := f"submissions/{args.identifier}-{datetime.now().strftime('%Y_%m_%d-%H_%M')}.txt"
            ),
            predicted_labels,
            fmt="%d",
        )
        timer(f"Time spent writing results on {file_path}")
