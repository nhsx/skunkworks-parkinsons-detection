"""
Training script to train the PD|Control classifier.

Subfolders are required to follow the convention:
"train/PD", "train/Control", "val/PD" and "val/Control".
"""
import os
from datetime import datetime

# ML includes
import torch
import torch.nn as nn
import torch.optim as optim

# NV AMP
from apex import amp
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

# Import our network
from polygeist.CNN.model import PDNet


def train_model(
    training_dump_path: str,
    model_dump_dir: str,
    batch_size: int = 32,
    num_epochs: int = 500,
) -> str:
    """
    Train the model using PD and Control training data.
    @arg training_dump_path: Base folder location of the training data
    @arg model_dump_dir: Directory where the model weights (checkpoint) should be saved
    @arg batch_size: Size of minibatch
    @arg num_epochs: Number of training epochs
    @return: name of the final checkpoint file
    """

    # We will set this within our training loop
    latest_model_name = None

    # Size we will need to down-sample to fit the model.
    input_size = 299

    # hyper-parameters
    learning_rate = 0.0125
    momentum = 0.9

    # Use cuda for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup tensorboard writer
    writer = SummaryWriter()

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(180),
                transforms.Resize(input_size),
                transforms.ToTensor(),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
            ]
        ),
    }

    # Create training and validation datasets
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(training_dump_path, x), data_transforms[x])
        for x in ["train", "val"]
    }

    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4
        )
        for x in ["train", "val"]
    }

    # Create our model
    model_ft = PDNet()
    # Send the model to GPU
    model_ft.to(device)
    #  Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=momentum)
    model_ft, optimizer_ft = amp.initialize(
        model_ft, optimizer_ft, opt_level="O1", verbosity=0
    )

    # Setup the loss func
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs-1}")
        print("==" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model_ft.train()  # Set model to training mode
            else:
                model_ft.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model_ft(inputs)
                    loss = criterion(
                        outputs.flatten(), labels.type("torch.cuda.HalfTensor")
                    )

                    # Greater than 0 will be class 1
                    preds = (outputs > 0).type(torch.HalfTensor)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        optimizer_ft.zero_grad()
                        # Amp trains about 3x faster than loss.backward
                        with amp.scale_loss(loss, optimizer_ft) as scaled_loss:
                            scaled_loss.backward()
                        optimizer_ft.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.flatten() == labels.data.cpu())

            epoch_loss = running_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders_dict[phase].dataset)

            # Write stats
            print(f"{phase} > Loss: {epoch_loss} Acc: {epoch_acc}")
            writer.add_scalar(f"loss/{phase}", epoch_loss, epoch)
            writer.add_scalar(f"accuracy/{phase}", epoch_acc, epoch)

            # deep copy the model
            if phase == "train":
                if epoch % 10 == 0:
                    latest_model_name = f"PDNET_checkpoint_{epoch}_{str(datetime.now().strftime('%H_%M_%S'))}"
                    model_ft.save_state(os.path.join(model_dump_dir, latest_model_name))
        print()

    return latest_model_name
