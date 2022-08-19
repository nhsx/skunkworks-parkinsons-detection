"""
This command line tool validates the classifier model and produces an ROC curve.
The purpose of this script is to validate the classifier
by producing summary performance statistics.
Validation slide data is used along with a model loaded
from a checkpoint file produced during training.
Instructions on how to run this file can be found in the README.md in this directory.
"""
import os
import errno
import argparse
import numpy as np
import matplotlib.pyplot as plt
from polygeist.validation import plot_roc, validate


def _ensure_exists(path: str) -> None:
    """
    Throw error if a given file or folder does not exist
    @arg path: path to file or folder
    """
    if not os.path.exists(path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)


def _validate_model() -> None:
    """Pass command line args to the train model function"""

    # Get arguments from command line
    parser = argparse.ArgumentParser(
        description="Validate classifier by producing ROC curve and performance statistics."
    )

    # Args to generate
    parser.add_argument(
        "--model_file",
        "-m",
        type=str,
        required=True,
        help="[str] Path to the model weight (checkpoint) data.",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=32,
        help="[int] Size of minibatches.",
    )
    parser.add_argument(
        "--training_dump_path",
        "-i",
        type=str,
        required=True,
        help="[str] Path to the training data ('val/<class>' subdirectory expected).",
    )
    parser.add_argument(
        "--roc_file",
        "-o",
        type=str,
        required=False,
        help="[str] Filename for the output ROC graph (.png).",
    )

    # Read arguments from the command line
    args = parser.parse_args()

    # Validity check arguments
    _ensure_exists(args.model_file)
    _ensure_exists(f"{args.training_dump_path}/val/Control")
    _ensure_exists(f"{args.training_dump_path}/val/PD")

    # Do validation
    output_data_and_labels = validate(
        args.model_file, args.training_dump_path, args.batch_size
    )

    # Calcuate summary statitics
    outputs = np.hstack(output_data_and_labels["outputs"])
    labels = np.hstack(output_data_and_labels["labels"])

    matched = outputs[labels == 1.0]
    non_matched = outputs[labels == 0]

    _, stats = plot_roc(
        plt, matched, non_matched, return_stats=True, verbose=False, steps=500000
    )

    # Display results
    specification_metric = "F1"
    in_ = np.where(stats[specification_metric] == np.max(stats[specification_metric]))[
        0
    ][0]
    print(
        f"Best M({specification_metric}): gives {stats['H'][in_]} hits and {stats['F'][in_]} FAs, S={stats['S'][in_]}, "
        f"P={stats['P'][in_]},"
        f" F1={stats['F1'][in_]}, A={stats['A'][in_]}"
    )

    # Save the ROC curve if requested
    if args.roc_file is not None:
        plt.savefig(args.roc_file)


if __name__ == "__main__":
    _validate_model()
