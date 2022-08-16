"""
Example script to show the classifier being used to highlight suspect areas.
"""
import os
import torch
from torchvision import transforms
import numpy as np
import lycon
from polygeist.CNN.model import PDNet


def _im_to_tensor(filename):
    """
    Loads an image to a torch tensor
    """
    # Load the file
    im = lycon.load(filename)
    # Make sure we have a three channel image
    assert im.shape[2] == 3
    # permute the channels from (y,x,c) to (c, y, x)
    return torch.tensor(im).permute(2, 0, 1)


def _chunk_tensor(tensor, chunk_size=512):
    """
    Generator to split the tensor into chunks.
    """
    _, yy, xx = tensor.shape
    # Tumble over the slice using a fixed window size
    for x in np.arange(0, xx - chunk_size, chunk_size):
        for y in np.arange(0, yy - chunk_size, chunk_size):
            # Note, this can be achieved by using tensor.unfold.
            # However, we are doing this to assure correctness.
            yield tensor[:, y : y + chunk_size, x : x + chunk_size].unsqueeze(0)


def _im_to_chunked_tensor(filename, chunk_size=512):
    """
    Creates a stacked tensor [Stack, Channels, Y, X],
    by consuming the chunk iterator on _im_to_tensor.
    """
    return torch.vstack(
        list(_chunk_tensor(_im_to_tensor(filename), chunk_size=chunk_size))
    )


def _im_to_chunked_iterator(filename, chunk_size=512, batches=100):
    """
    Iterates over a stacked list of chunks, with an aim of n batches
    """
    batches = _im_to_chunked_tensor(filename, chunk_size=chunk_size).split(batches)
    for batch in batches:
        yield batch


def label_image_with_confidence(
    model_file: str,
    file_to_stain: str,
    output_path: str,
    threshold=20,
    pc=0.005,
    chunk_size=512,
) -> None:
    """
    Annotate the stained slide and save to disk.
    @arg model_file: path to the model weight data checkpoint (str)
    @arg file_to_stain: path to the original svs file (str)
    @arg output_path: path where the image should be saved (str)
    @arg threshold: value same as used in synthetic staining
    @arg pc: value same as used in synthetic staining
    @arg chunk_size: size of patch in pixels
    """
    # We redefine our transforms based on the image already being a torch tensor
    input_size = 299
    downsample_for_tensor = transforms.Compose([transforms.Resize(input_size)])

    # Create our model
    model_ft = PDNet()

    # Apply state
    model_ft.apply_state(model_file)

    # Send the model to GPU and set in eval mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ft.to(device)
    model_ft = model_ft.eval()
    model_ft = model_ft.half()

    # in evaluation mode
    with torch.set_grad_enabled(False):
        # list of results which we will pass to marking function
        results = []
        # Chunk through our image in chunks of 512px
        for batch in _im_to_chunked_iterator(file_to_stain, chunk_size=512, batches=50):
            # Apply downsample
            downsampled = downsample_for_tensor(batch)
            # Format as float
            formatted = downsampled.type(torch.cuda.HalfTensor).to(device) / 255.0
            # Append with results of the batch
            results.append(model_ft(formatted))

    # make a 1-d array of results
    flat_results = np.hstack(np.vstack([x.tolist() for x in results]))

    image = lycon.load(file_to_stain)
    # Get the height and width of the slice
    xx, yy, _ = image.shape
    # Tumble over the slice using a fixed window size
    c = 0
    for x in np.arange(0, xx - chunk_size, chunk_size):
        for y in np.arange(0, yy - chunk_size, chunk_size):
            # Calculate the difference in our green channel
            section = image[x : x + chunk_size, y : y + chunk_size, :]
            diff = np.abs(
                section[:, :, 0].astype(float) - section[:, :, 1].astype(float)
            )
            # Test to see if this is a sufficiently stained section
            # - these params (threshold and pc) need to be
            # the same as used in the synthetic staining routine
            if np.sum(diff > threshold) / (chunk_size**2) > pc:
                if flat_results[c] > 0.95:
                    section[:, 0:10, 0] = 255
                    section[0:10, :, 0] = 255
                    section[-10:-1, :, 0] = 255
                    section[:, -10:-1, 0] = 255
            image[x : x + chunk_size, y : y + chunk_size, :] = section
            # tick up our array counter
            c += 1
    lycon.save(f"{output_path}/{os.path.basename(file_to_stain)}", image)
