"""
This command line tool generates fake slide datasets (.png) for PD and Control classes.
The purpose of this script is to synthesize slide data
so the use or real data can be avoided.
Multiple files are created that can then be used for training and validation.
Instructions on how to run this file can be found in the README.md in this directory.
"""
import os
import errno
import argparse
import imageio as io
import numpy as np
from polygeist.data_faker import generate_fake_slide_data


def _ensure_exists(path: str) -> None:
    """
    Throw error if a given file or folder does not exist
    @arg path: path to file or folder
    """
    if not os.path.exists(path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)


def _generate_fake_slide_dataset() -> None:
    """Pass command line args to the colourise and segment function"""

    # Get arguments from command line
    parser = argparse.ArgumentParser(
        description="Create synthetically stained slide data (.svs -> .pngs)"
        + " with tile segments filtered on colour."
    )

    # Args to generate
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        required=True,
        help="[str] Path to the output folder where slides (.pngs) are to be created).",
    )
    parser.add_argument(
        "--seed_start",
        "-s",
        type=int,
        required=True,
        help="[int] Value of first seed.",
    )
    parser.add_argument(
        "--quantity",
        "-q",
        type=int,
        required=True,
        help="[int] Number of slides (.pngs) to be generated.",
    )
    parser.add_argument(
        "--is_control",
        default=False,
        action="store_true",
        help="Whether the slide mimicks Control or PD features",
    )

    # Read arguments from the command line
    args = parser.parse_args()

    # Validity check arguments
    _ensure_exists(args.output_path)

    # Generate slides
    for seed in range(args.seed_start, args.seed_start + args.quantity):

        # Generate pseudo-slides
        slide = generate_fake_slide_data(
            seed=seed, a_prob=0.015 if args.is_control else 0.15
        )

        # Write the slide
        io.imwrite(
            f"{args.output_path}/slide_{seed}.png",
            slide.astype(np.uint8),
        )


if __name__ == "__main__":
    _generate_fake_slide_dataset()
