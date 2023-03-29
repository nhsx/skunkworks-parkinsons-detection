"""
Preprocessing script to synthetically stain and segment a slide.
"""
import os
import imageio as io

# Import Slide interface wrapper
from polygeist.slidecore import AperioSlide, SyntheticSlide

# Import Staining libraries
from ext.ideepcolor.data.colorize_image import ColorizeImageTorch as ColouriseImageTorch
import ext.ideepcolor.models.pytorch.model as model

# ML includes
import torch

# Numeric includes and plotting
import numpy as np
from skimage.transform import resize

# Configuration
state_path = os.path.join(os.path.dirname(__file__), '..', 'ext/ideepcolor/models/pytorch/caffemodel.pth')

"""Location of the synthetic staining model weights"""

staining_window = 512
"""The size we would like to stain at one time"""

network_roi_size = 256
"""Network ROI, staining_window will be transformed to this size and back again"""


# Preprocessing function
def colourise_slide_and_segment(
    slide_file: str,
    is_synthetic: bool,
    dump_path_full: str,
    dump_path_segmented: str,
    subdirectory: str,
) -> None:
    """
    Preprocess the slide.
    @arg slide_file: path to the slide to be processed (str),
    @arg is_synthetic: whether the slide is real or fake (bool),
    @arg dump_path_full: directory where full image should be saved (str)
    @arg dump_path_segmented: directory where filtered images should be saved (str)
    @arg subdirectory: directory append (i.e. class, either PD or Control) (str)
    """
    # Run model on gpu this is for the colourisation network functions only
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize colourisation class, torch model
    colour_model = ColouriseImageTorch(Xd=network_roi_size)

    # Pretrained model, the library uses the same model (caffe)
    # for its caffe and pytorch implementations
    colour_model.prep_net(0, state_path, SIGGRAPHGenerator=model.SIGGRAPHGenerator)

    # We will use our 2um (4um^2) pixels, as we used in the jpeg processing tests
    pixel_width_in_microns = 0.504 * 4

    slide = SyntheticSlide(slide_file) if is_synthetic else AperioSlide(slide_file)
    slide_at_um = slide.get_slide_with_pixel_resolution_in_microns(pixel_width_in_microns)

    # Set the slide window size to correspond with our network,
    # we will use  a broader window, which we used during some initial testing
    slide.window_size = staining_window

    # colour threshold (for staining)
    threshold = 20
    pc = 0.005

    # Get the dimensions of our slide
    yy, xx, _ = slide_at_um.shape

    # basename for dumping out
    base = os.path.basename(slide_file)

    # counter for our dump
    c = 0

    # output array for dumping to disk
    output_c = np.zeros((yy, xx, 3), dtype=float)
    for x in np.arange(0, xx - slide.window_size, slide.window_size):
        for y in np.arange(0, yy - slide.window_size, slide.window_size):

            # Get our ROI for this pass in the network
            roi = slide_at_um[y:y + slide.window_size, x:x + slide.window_size, :]

            # Here we are producing our 'suggestion' mask for the network,
            # we mask values which are stained brown
            # (negative mask is therefore > a blue minimum value).
            b_mask_threshold = 50
            roi_mask = roi[:, :, 2] > b_mask_threshold

            # reshape the mask for the network
            roi_input = resize(roi_mask, (256, 256))

            # Clean up the mask,
            # format its dimensions to have a batch size of 1 and send it to our torch device
            cleaned_mask: np.ndarray = ~(roi_input == 0)
            mask = torch.Tensor(cleaned_mask.astype(float)).unsqueeze(0).to(device)

            # These are La*b* coordinates,
            # we use them as a negative is positive (cookie-cut) mask,
            # they will tell our network we want regions similar to those hinted
            # (and their surrounding areas) to be stained with these coordinates.
            # These coordinates allow post-hoc rotation in La*b* colour space.
            a = -50
            b = 63
            ab = np.zeros((2, 256, 256))
            ab[0, ~cleaned_mask] = a
            ab[0, ~cleaned_mask] = b

            # To the GPU!
            input_ab = torch.Tensor(ab.astype(float)).to(device)

            # Ok, now we run the network on the image
            colour_model.set_image(roi)
            colour_model.net_forward(input_ab, mask)  # run model, returns 256x256 image

            # get mask, input image,
            # and result in full resolution using the network functions
            img_out_fullres = (
                colour_model.get_img_fullres()
            )  # get image at full resolution

            output_c[y:y + slide.window_size, x:x + slide.window_size, :] = img_out_fullres.copy()

            # Calculate the difference in our green channel,
            # and only dump portions of interest
            diff = np.abs(
                img_out_fullres[:, :, 0].astype(float)
                - img_out_fullres[:, :, 1].astype(float)
            )
            # if half of 1% of the slide is over our threshold,
            # we will include the slide.
            if np.sum(diff > threshold) / (slide.window_size**2) > pc:
                # Should we dump the slide?
                # (see if there are any highlighted bodies in the slide)
                io.imwrite(
                    f"{dump_path_segmented}/{subdirectory}/{base}_{c}.png",
                    255.0 - img_out_fullres,
                )
                c += 1

    # It is easier to identify the staining when we flip to dark mode.
    # Invert the image.
    io.imwrite(
        f"{dump_path_full}/{subdirectory}/{base}_synthetic_stain.png", 255.0 - output_c
    )
