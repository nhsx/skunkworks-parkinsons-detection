"""
This command line tool trains the classifier.
The purpose of this script is to train the classifier
to distingish between Control and PD slide patches.
Model training weights are generated from 'train'
and validated using 'val' subdirectories.
Instructions on how to run this file can be found in the README.md in this directory.
"""
import os
import errno
import argparse
from polygeist.training import train_model


def _ensure_exists(path: str) -> None:
    """
    Throw error if a given file or folder does not exist
    @arg path: path to file or folder
    """
    if not os.path.exists(path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)


def _train_model() -> None:
    """Pass command line args to the train model function"""

    # Get arguments from command line
    parser = argparse.ArgumentParser(
        description="Train classifier to distinguish between"
        + " Control and PD slide patches."
    )

    # Args to generate
    parser.add_argument(
        "--training_dump_path",
        "-i",
        type=str,
        required=True,
        help="[str] Path to the training data ('train/<class>' and"
        + " 'val/<class>' subdirectories expected).",
    )
    parser.add_argument(
        "--model_dump_dir",
        "-o",
        type=str,
        required=True,
        help="[str] Path to the output folder"
        + " (where model weight checkpoints will be saved).",
    )

    # Read arguments from the command line
    args = parser.parse_args()

    # Validity check arguments
    _ensure_exists(f"{args.training_dump_path}/train/Control")
    _ensure_exists(f"{args.training_dump_path}/train/PD")
    _ensure_exists(f"{args.training_dump_path}/val/Control")
    _ensure_exists(f"{args.training_dump_path}/val/PD")
    _ensure_exists(args.model_dump_dir)

    # Do training
    latest_model_name = train_model(args.training_dump_path, args.model_dump_dir)
    print(f"Final model saved as: {latest_model_name}")


if __name__ == "__main__":
    _train_model()
