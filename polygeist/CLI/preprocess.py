"""
This command line tool preprocesses svs slide data before training and validation.
The purpose of this script is to synthetically stain slide data and filter the results.
A full size image is created along with any tile segments that pass a colour filter.
Instructions on how to run this file can be found in the README.md in this directory.
"""
import os
import errno
import argparse
from polygeist.preprocess import colourise_slide_and_segment


def _ensure_exists(path: str) -> None:
    """
    Throw error if a given file or folder does not exist
    @arg path: path to file or folder
    """
    if not os.path.exists(path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)


def _process_slide() -> None:
    """Pass command line args to the colourise and segment function"""

    # Get arguments from command line
    parser = argparse.ArgumentParser(
        description="Create synthetically stained slide data"
        + " (.svs -> .pngs) with tile segments filtered on colour."
    )

    # Args to generate
    parser.add_argument(
        "--input_slide",
        "-i",
        type=str,
        required=True,
        help="[str] Path to the input slide (.svs).",
    )
    parser.add_argument("--is_synthetic", default=False, action="store_true")
    parser.add_argument(
        "--output_path_full",
        "-of",
        type=str,
        required=True,
        help="[str] Path to the output folder"
        + " (where full size .pngs are to be created).",
    )
    parser.add_argument(
        "--output_path_segmented",
        "-os",
        type=str,
        required=True,
        help="[str] Path to the output folder"
        + " (where segmented and filtered .pngs are to be created).",
    )
    parser.add_argument(
        "--subdirectory",
        "-s",
        type=str,
        required=True,
        help="[str] Subdirectory of the output folder (i.e. the Class: Control or PD)",
    )

    # Read arguments from the command line
    args = parser.parse_args()

    # Validity check arguments
    _ensure_exists(args.input_slide)
    _ensure_exists(f"{args.output_path_full}/{args.subdirectory}")
    _ensure_exists(f"{args.output_path_segmented}/{args.subdirectory}")

    # Do processing
    colourise_slide_and_segment(
        args.input_slide,
        args.is_synthetic,
        args.output_path_full,
        args.output_path_segmented,
        args.subdirectory,
    )


if __name__ == "__main__":
    _process_slide()
