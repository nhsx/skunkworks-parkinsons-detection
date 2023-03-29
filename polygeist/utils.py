import os
from typing import Iterable, Tuple

import imageio as io
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
import json
import fnmatch
import pathlib
import shutil

CONTROL = "C"
AD = "AD"
PD = "PD"
MSA = "MSA"


def make_path(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def copy_file(source, dest):
    shutil.copyfile(source, dest)


class SegmentationFilesDirectoryHandler:
    def __init__(self, root, filetype=".jpg"):
        self.root = root
        # Where are we putting copies of the data for training and test
        self.training_dump_path = root + "/partitioned_data/"
        self.training_dir_name = "train"
        self.validation_dir_name = "val"
        self.conditions = None
        self.filetype = filetype

    def make_train_and_validation_folders_for_conditions(self, conditions=None):

        if conditions is None:
            self.conditions = ["Controls", "PD"]
        else:
            self.conditions = conditions
        # Make the root path
        make_path(self.training_dump_path)
        # Make the condition paths
        for run in [self.training_dir_name, self.validation_dir_name]:
            for cond in self.conditions:
                make_path(f"{self.training_dump_path}/{run}/{cond}/")

    def split_and_copy_root_data_to_train_and_validation(
        self,
        case_filter_for_train=None,
        include_filter=None,
        exclude_filter=None,
        condition=None,
        slide_index_filter=None,
        training=False,
    ):
        files_in_root = glob(f"{self.root}/*{self.filetype}")
        for file in files_in_root:
            # Get the slide and case ID
            case, slide = get_case_and_slide(file)

            # Check the slide index filter
            if slide_index_filter:
                if slide not in slide_index_filter:
                    continue

            # Get the base filename
            base = os.path.basename(file)
            # Load the file to check its dims
            im = io.imread(file)
            # if it does not have the same x & y we will skip
            if im.shape[0] != im.shape[1]:
                continue

            # If there is a case filter, continue if the case is not in the case filter
            if case_filter_for_train:
                if case not in case_filter_for_train:
                    continue
            # if the include filter is set, and the case should not be included, then continue
            if include_filter:
                if include_filter not in file:
                    continue

            # If the case should be excluded then exclude it.
            if exclude_filter:
                if exclude_filter in file:
                    continue

            # Change the target dir by training flag
            target_dir = (
                self.training_dir_name if training else self.validation_dir_name
            )

            # Copy the file to the training dir
            copy_file(
                file, f"{self.training_dump_path}/{target_dir}/{condition}/{base}"
            )


def plot_roc(
    matched: np.ndarray,
    non_matched: np.ndarray,
    label="ROC",
    title="ROC Curve",
    axes=None,
    steps=1000,
    verbose=False,
    max_value=None,
):
    """
    Plot an ROC curve for the given matched and non-matched scores
    :param matched: A numpy array of matched scores
    :param non_matched: A numpy array of non-matched scores
    :param label: The label for the plotted line
    :param title: The title for graph
    :param axes: The axes to use for the plot; if None, new axes will be made
    :param steps: The number of steps to use for the plot (# bins)
    :param verbose: When true, output information about threshold values
    :param max_value: Use this to override the max score to consider (useful for zooming the bins on the low FA area)
    :return: The axes used for the plot
    """
    if axes is None:
        axes = plt.axes()
    h = []
    fa = []
    # Plot a ROC for <steps> possible thresholds from 0 .. max score
    for hits, false_alarms, a, p, s, f1 in evaluate_threshold_sweep(
        steps, matched, non_matched, verbose, max_value
    ):
        h.append(hits)
        fa.append(false_alarms)
    h = np.array(h)
    fa = np.array(fa)
    axes.plot(fa, h, label=label)
    axes.set_xlabel("False Alarm Rate")
    axes.set_ylabel("Hit Rate")
    axes.legend()
    axes.set_title(title)
    return axes


def evaluate_threshold_sweep(
    steps: int,
    matched: np.ndarray,
    non_matched: np.ndarray,
    verbose: bool = False,
    max_value: float = None,
) -> Iterable[Tuple[float, float, float, float, float, float]]:
    """
    Sweep through C{steps} possible thresholds, between 0 and the max value in the distribution (or C{max_score}, if
    given), and evaluate the hit and false alarm rate at each point. Yields a generator of hit-rate (recall/TPR),
    false positive rate, accuracy, precision, sensitivity, and f1 score
    @arg steps: The number of thresholds to test
    @arg matched: The distribution of match scores
    @arg non_matched: The distribution of non-match s ores
    @arg verbose: Set to C{True} to make this method print out the score information for each threshold
    @arg max_value: Set to any float to override the maximum score value derived from the match/non-match distributions
    """
    if max_value is None:
        max_value = max(matched.max(), non_matched.max())
    for i in range(0, steps):
        test_threshold = (i / steps) * max_value
        tpr = np.count_nonzero(matched > test_threshold) / matched.size
        tnr = np.count_nonzero(non_matched < test_threshold) / non_matched.size
        fpr = np.count_nonzero(non_matched > test_threshold) / non_matched.size
        accuracy = (tpr + (1.0 - fpr)) / 2
        if tpr > 0 and fpr > 0:
            precision = tpr / (tpr + fpr)
            s = tnr / (tnr + fpr)
        else:
            precision = -1
            s = -1
        if precision > 0 and tpr > 0:
            f1 = (2 * precision * tpr) / (precision + tpr)
        else:
            f1 = -1
        if verbose:
            print(
                f"Step:{i}: Thresh {test_threshold} gives {tpr} hits and {fpr} FAs, "
                f"S={s}, P={precision}, F1={f1}, A={accuracy}"
            )
        yield tpr, fpr, accuracy, precision, s, f1


def collect_cases(config):
    # Collect the positive and negative cases
    positive_cases = []
    for p in glob(
        config["root_directory"] + config["positive_case_folder"] + "/*/",
        recursive=True,
    ):
        # Get the case ID
        case = os.path.basename(os.path.normpath(p))
        positive_cases.append(case)

    negative_cases = []
    for p in glob(
        config["root_directory"] + config["negative_case_folder"] + "/*/",
        recursive=True,
    ):
        # Get the case ID
        case = os.path.basename(os.path.normpath(p))
        negative_cases.append(case)

    return positive_cases, negative_cases


def calc_median_score_list(cases, path, rgb_threshold=20):
    # Simple median based classifier
    score_list = []
    for case in cases:
        count_list = []
        for p in glob(path, recursive=True):
            if case in p:
                im = io.v2.imread(p)
                count_list.append(np.sum(im[:, :, 2] < rgb_threshold))
        if len(count_list) > 0:
            score_list.append(np.median(count_list))
        else:
            score_list.append(0)
    return score_list


def region_count_score_list(cases, results):
    # Count for comparison with simple median classifier
    score_list = []
    for case in cases:
        count_list = []
        for k, v in results.items():
            if case in k:
                nv = v.cpu().numpy()[0][0]
                count_list.append(nv)
        if len(count_list) > 0:
            score_list.append(np.median(count_list))
        else:
            pass
    return score_list


def load_filenames_and_generate_conditions(list_filename):
    # Conditions
    with open(list_filename, "r") as file:
        lines = file.readlines()

    case_dict = {}
    for f in lines:
        case, slide_index = get_case_and_slide(f)
        condition = (
            CONTROL if CONTROL in f else (AD if AD in f else (MSA if MSA in f else PD))
        )
        if case not in case_dict:
            case_dict[case] = condition
    return case_dict


# An oversight during processing; we didn't put the file name in the json, as we auto processed the slides using
# their filename, so here we just parse the names to get the case and slide
def get_case_and_slide(fname):
    f = os.path.basename(fname)
    case = f.split("-")[0]
    for erroneous in ["a", "b", "+", "A", "B"]:
        f = f.replace(erroneous, "")
    slide = int(f.split("-")[1].split("_")[0])
    return case, slide


def load_jsons_from_directory(directory):
    slide_parsed_jsons = {}
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, "*.json"):
            with open(os.path.join(root, filename), "r") as read_file:
                data = json.load(read_file)
                slide_parsed_jsons[filename] = data
    return slide_parsed_jsons
