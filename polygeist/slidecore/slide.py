"""
Wrapper around slideio to unify data import.
"""
import slideio
import numpy as np
import lycon
from skimage.transform import resize
from abc import ABC, abstractmethod


class Slide(ABC):
    """Slide abstract base class to allow consistent access to real and synthetic data."""

    def __init__(self):
        """Instantiate the Slide object (load from a file)."""

        self.window_size = 299
        """Window size for iterating (299)"""

    def _calc_size(self, x: int, y: int, microns: float):
        """
        Calculate final image size based on the size of the pixels
        @arg x: Original size x (int)
        @arg y: Original size y (int)
        @arg microns: Size of pixels
        @return: Scaled dimensions (int, int)
        """
        factor = microns / self._px_um
        return (int(np.ceil(x / factor)), int(np.ceil(y / factor)))

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
        self._synthetic_data = lycon.load(filename)

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


class AperioSlide(Slide):
    """Slide object to allow consistent access to real data."""

    def __init__(self, filename):
        super().__init__()
        # populate with real data
        slide_handle = slideio.open_slide(filename, "SVS")
        self._root_scene = slide_handle.get_scene(0)
        # pixel width in microns
        self._px_um = 0.5040

    def get_slide_with_pixel_resolution_in_microns(self, microns=0.5040):
        # Get the height and width of the slice
        _, _, xx, yy = self._root_scene.rect
        size = self._calc_size(xx, yy, microns)
        return self._root_scene.read_block(size=size)
