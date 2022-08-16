"""
Validation script to measure and report the classifier's performance.

Subfolders are required to follow the convention:
"val/PD" and "val/Control".
"""
import os
import torch
from torchvision import datasets, transforms
import numpy as np
from tqdm.notebook import tqdm
from polygeist.CNN.model import PDNet


def plot_roc(
    plt,
    matched: np.ndarray,
    non_matched: np.ndarray,
    label="ROC",
    axes=None,
    steps=1000,
    verbose=False,
    max_value=None,
    return_stats=False,
):
    """
    Plot an ROC curve for the given matched and non-matched scores
    @arg plt: matplotlib reference object
    @arg matched: A numpy array of matched scores
    @arg non_matched: A numpy array of non-matched scores
    @arg label: The label for the plotted line
    @arg axes: The axes to use for the plot; if None, new axes will be made
    @arg steps: The number of steps to use for the plot (# bins)
    @arg verbose: When true, output information about threshold values
    @arg max_value: Use this to override the max score to consider
    @arg return_stats: Return a dictionary with performance metrics
    @return: The axes used for the plot
    """
    if axes is None:
        axes = plt.axes()
    h = []
    fa = []
    stats = {
        "A": [],
        "P": [],
        "S": [],
        "F1": [],
        "H": [],
        "F": [],
    }
    # Plot a ROC for <steps> possible thresholds from 0 .. max score
    if max_value is None:
        max_value = max(matched.max(), non_matched.max())
    for i in range(0, steps):
        test_threshold = (i / steps) * max_value
        hits = np.count_nonzero(matched > test_threshold) / matched.size
        tn = np.count_nonzero(non_matched < test_threshold) / non_matched.size
        false_alarms = np.count_nonzero(non_matched > test_threshold) / non_matched.size
        A = (hits + (1.0 - false_alarms)) / 2
        if hits > 0 and false_alarms > 0:
            P = hits / (hits + false_alarms)
            S = tn / (tn + false_alarms)
        else:
            P = -1
            S = -1
        if P > 0 and hits > 0:
            F1 = (2 * P * hits) / (P + hits)
        else:
            F1 = -1
        if verbose:
            print(
                f"Step:{i}: Thresh {test_threshold} gives {hits} hits and"
                f" {false_alarms} FAs, S={S}, P={P}, F1={F1}, A={A}"
            )
        h.append(hits)
        fa.append(false_alarms)

        # Stats
        stats["A"].append(A)
        stats["P"].append(P)
        stats["S"].append(S)
        stats["F1"].append(F1)

    # Stats
    stats["H"].append(h)
    stats["F"].append(fa)

    for k, v in stats.items():
        stats[k] = np.array(v).flatten()

    # Numpy
    h = np.array(h)
    fa = np.array(fa)
    # Plots
    axes.plot(fa, h, label=label)
    axes.set_xlabel("False Alarm")
    axes.set_ylabel("Hits")
    axes.legend()
    axes.set_title("ROC Curve")

    if return_stats:
        return axes, stats
    return axes


def _create_dataloader(training_dump_path: str):
    """
    Create a torch 'DataLoader' for the validation data.
    @arg training_dump_path: Base folder location of the training data
    @return: torch dataloader
    """
    # Data normalization for validation
    input_size = 299

    downsample = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ]
    )

    batch_size = 32

    # Create validation dataset
    image_dataset = datasets.ImageFolder(
        os.path.join(training_dump_path, "val"), downsample
    )

    # Create validation dataloader
    return torch.utils.data.DataLoader(
        image_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )


def _t2n(tensor, apply=None):
    """
    Tensor to Numpy
    When using Torch, converting to Numpy often errors with the need to pass back to CPU
    first and sometimes additionally detaching explicitly.
    tolist() does this automatically, and works when exclusively on CPU.
    So we go to list, then explicity to numpy
    """
    return np.array(tensor.tolist())


def validate(model_file: str, training_dump_path: str) -> dict:
    """
    Stand the model up and produce validation metrics.
    @arg model_file: path to weights (checkpoint) to be loaded (str)
    @arg training_dump_path: Base folder location of the training data
    @return: output data and labels (dict)
    """
    # Create our model
    model_ft = PDNet()

    # Apply state
    model_ft.apply_state(model_file)

    # Send the model to GPU and set in eval mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ft.to(device)
    model_ft = model_ft.eval()

    # Create dataloader for validation data
    dataloader = _create_dataloader(training_dump_path)

    # Where we will store our outputs and data
    output_data_and_labels = {"outputs": [], "labels": []}

    # Validate the model using the validation set, and a threshold sweep
    for inputs, labels in tqdm(dataloader):
        # Iterate over data.
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model_ft(inputs)
            output_data_and_labels["outputs"].append(np.hstack(_t2n(outputs)))
            output_data_and_labels["labels"].append(_t2n(labels))

    return output_data_and_labels
