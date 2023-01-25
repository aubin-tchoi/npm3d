import argparse
from typing import Tuple

import numpy as np

from ply import read_ply


def parse_args() -> argparse.Namespace:
    """
    Parses the command line arguments. Also produces the help message.
    """
    parser = argparse.ArgumentParser(description="Launches runs of ICP")

    parser.add_argument(
        "--rms_threshold",
        type=float,
        default=1e-4,
        help="RMS threshold used as a stopping criteria",
    )
    parser.add_argument(
        "--max_iter", type=int, default=30, help="Limit on the number of iterations"
    )
    parser.add_argument(
        "--skip_rigid",
        action="store_false",
        help="Skip the rigid transformation example on bunnies",
    )
    parser.add_argument(
        "--skip_2d",
        action="store_false",
        help="Skip ICP on the 2D example",
    )
    parser.add_argument(
        "--skip_bunnies",
        action="store_false",
        help="Skip ICP on the example on bunnies",
    )
    parser.add_argument(
        "--skip_nddc",
        action="store_false",
        help="Skip ICP on the Notre Dame des Champs example",
    )

    return parser.parse_args()


def get_data(data_path: str, ref_path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = read_ply(data_path)
    ref = read_ply(ref_path)

    return (
        np.vstack((data["x"], data["y"], data["z"])),
        np.vstack((ref["x"], ref["y"], ref["z"])),
    )
