"""
Wrapper around slideio to unify data import.
"""
import slideio
import numpy as np
import imageio.v2 as io
from skimage.transform import resize
from abc import ABC, abstractmethod
from .calibration import CalibrationProfile


class Slide(ABC):
    """Slide abstract base class to allow consistent access to real and synthetic data."""

    def __init__(self):
        """Instantiate the Slide object (load from a file)."""

        self.window_size = 299
        # Window size for iterating (299)
        self.calibration_profile = None
        # Default pixel size in um - should always be overridden in children
        self._px_um = 1

    def _calc_size(self, x: int, y: int, microns: float):
        """
        Calculate final image size based on the size of the pixels
        @arg x: Original size x (int)
        @arg y: Original size y (int)
        @arg microns: Size of pixels
        @return: Scaled dimensions (int, int)
        """
        factor = microns / self._px_um
        return int(np.ceil(x / factor)), int(np.ceil(y / factor))

    @abstractmethod
    def get_slide_with_pixel_resolution_in_microns(self, microns=0.5040):
        """
        Return the slice at the given resolution.
        @arg microns: Resolution (float)
        @return: The resampled block from the slice (np.array)
        """
        pass


class SyntheticSlide(Slide):
    """Slide object to allow consistent access to synthetic data."""

    def __init__(self, filename):
        super().__init__()
        # populate with synthetic data
        self._synthetic_data = io.imread(filename)
        if self._synthetic_data is None:
            raise IOError(f"Error loading {filename}")

        # In native slide format our pixels are always .5um, to
        # make all the processing realistic we need to determine our
        # synthetic um based on its width (for the entire slide vs real)
        typical_imaging_span = 57768
        scale = self._synthetic_data.shape[1] / typical_imaging_span
        self._px_um = 0.5040 / scale

    def get_slide_with_pixel_resolution_in_microns(self, microns=0.5040):
        # Get the height and width of the slice
        yy, xx, _ = self._synthetic_data.shape
        size = self._calc_size(xx, yy, microns)
        return (resize(self._synthetic_data, size) * 255.0).astype(np.uint8)


class SpectralSlideGenerator:
    def __init__(self, width=100, height=100, filename=None, control=False):
        self.responses = np.array([[7.10357511, 12.61506218, 13.59695489],
                                   [8.81961536, 10.18302502, 5.08567669],
                                   [11.02074647, 15.41804066, 12.77042165],
                                   [17.05035857, 17.64819458, 9.17788779],
                                   [0.45574971, 4.32897163, 1.68161384]])

        image = np.zeros((height, width, 3))
        image[:, :, 0] = (np.random.random((height, width)) * 4).astype(int)

        # turn all DAB into BT and then spatter some DAB
        if control:
            image[image[:, :, 0] == 0, 0] = 3
            image[np.random.random((height, width)) > .95, 0] = 0

        for i in np.arange(0, 5):
            image[image[:, :, 0] == i, :] = self.responses[i, :]

        mulfield = np.random.random((height, width)) * 10
        scalar = np.zeros((height, width, 3))
        for i in np.arange(0, 3):
            scalar[:, :, i] = mulfield
        self.image = np.multiply(image, scalar)

        if filename:
            io.imwrite(filename, self.image.astype(np.uint8))


class AperioSlide(Slide):
    """Slide object to allow consistent access to real data."""

    def __init__(self, filename, calibrate_image=False):
        super().__init__()
        # populate with real data
        slide_handle = slideio.open_slide(filename, "SVS")
        # Unfortunately slideio has a bug which prevents failure checking

        self._root_scene = slide_handle.get_scene(0)
        # pixel width in microns
        self._px_um = 0.5040

        # Calibration profile, to convert from Sensor RGB to sRGB or other colour space.
        self.calibration_profile = CalibrationProfile(filename) if calibrate_image else None

    def get_slide_with_pixel_resolution_in_microns(self, microns=0.5040):
        # Get the height and width of the slice
        _, _, xx, yy = self._root_scene.rect
        size = self._calc_size(xx, yy, microns)

        # Grab the uncalibrated image from the slide file
        uncalibrated_slice = self._root_scene.read_block(size=size)

        # Return the uncalibrated file, if no calibration profile is provided
        if self.calibration_profile is None:
            return uncalibrated_slice

        # return the calibrated image
        return self.calibration_profile.convert_to_srgb(uncalibrated_slice)
