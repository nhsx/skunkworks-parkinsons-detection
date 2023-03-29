import traceback

import tifffile
from PIL import ImageCms, Image
import io
import numpy as np


def generate_icc_to_srgb_transform(icc_profile):
    """
    Return colour transformation object
    @arg icc_profile: icc_profile object (PIL ImageCMS)
    @return: PIL transform (PIL.ImageCMS.ImageCMSTransform)
    """
    if icc_profile is None:
        return None

    """ The standard colour profile for the standardised (device independent) RGB colour space """
    sRGB_profile = ImageCms.createProfile("sRGB")

    """ Build the icc to SRGB transform and return it"""
    return ImageCms.buildTransformFromOpenProfiles(
        icc_profile, sRGB_profile, "RGB", "RGB"
    )


def get_icc_profile_from_slide(filename):
    """
    Return an ICC profile from a valid slide file
    @arg filename: path to pathology slide (string)
    @return: ICC profile (PIL.ImageCMS.ImageCMSProfile)
    """

    """ Slides are usually glorified TIFF files, while tifffile might not be able to decode the image
        data, it should be able to decode the header file, where the ICC profile resides.  Aperio slides load
        correctly.
    """
    try:
        with tifffile.TiffFile(filename) as tiff:
            for page in tiff.pages:
                for tag in page.tags:
                    if "ColorProfile" in tag.name:
                        # It is conceivable that each layer could have a different profile, but in reality if we see
                        # a tag with a color profile, we should expect every layer in that image to have been
                        # taken with the same microscope, at the same time, with the same profile.  So we will
                        # just return that as a PIL Colour Management System Profile.
                        return ImageCms.ImageCmsProfile(io.BytesIO(tag.value))
    except RuntimeError:
        traceback.print_exc()
        return None

    # Default, return None
    return None


class CalibrationProfile:
    """Chromatic calibration profile, representing the sensor information from a device"""

    def __init__(self, filename=None):
        self.base_icc_profile = get_icc_profile_from_slide(filename)
        """ The loaded profile from a slide/image for the purpose of building a colour transform """
        self.microscope_to_sRGB_transform = generate_icc_to_srgb_transform(
            self.base_icc_profile
        )

    def convert_to_srgb(self, numpy_image):
        """
        Converts a numpy image from sensor RGB to sRGB
        @arg numpy_image: a numpy array of a slide (numpy array)
        @return: numpy_image (in place) (numpy array)
        """
        # This should not encourage a copy, however we should test it.
        pil_view_on_slide = Image.fromarray(numpy_image)
        # This allows PIL to modify the array without a copy.
        pil_view_on_slide.readonly = False
        # Apply the transformation from sensor RGB to sRGB in place.
        ImageCms.applyTransform(
            pil_view_on_slide, self.microscope_to_sRGB_transform, inPlace=True
        )

        # Suppress type error as PIL Image is iterable as something `array_like`
        # noinspection PyTypeChecker
        return np.asarray(pil_view_on_slide)
