import fnmatch
from polygeist.slidecore import AperioSlide, SyntheticSlide, SpectralSlideGenerator
import numpy as np
import imageio as io
import os
import cv2
import matplotlib
import json
import pathlib


def generate_densities_and_map_coordinates(raw_image, stride, raw_threshold, class_threshold):
    """
        Produces density maps for a given decomposed image, and areas above a signal and classification threshold

        Definitions
        ===========
        Definitions of terms used in procedure below::
            I : [DAB Image],  i : [rows of image], j: [columns of image]
            D : [Density Map], s: [Stride], si: [i / stride], sj: [j / stride]
            B : [Binary map], T : [Signal Threshold], C: [Classification Threshold]
            mu : [Array mean], R : [ROI Array], O: [Coordinates array], Om: [Map Coordinates Array]

        Procedure
        =========
        Pseudocode for procedure generating density maps::
            B(i,j) = I(i,j) < T
            D(si, sj) = mu(I(i:i+s, j:j+s))
            R(si, sj) = D > C
            Om = (si, sj) for si and sj if R(si, sj)
            O = (i, j) for i and j if R(i / stride, j / stride)

        @arg raw_image: NxMxF spectral image, where F is the number of channels (Fd np.array)
        @arg stride: The window size over which we will tumble
        @arg raw_threshold: The value per pixel to be considered a 'vote' for that pixel containing significant signal
        @arg class_threshold: The mean density required for that region to be considered an ROI
        @return: NxM densities array, coordinates in the image of ROIs, coordinates in the map of ROIs (np.arrays)
    """

    # Coordinates for output
    coordinates = []
    map_coordinates = []

    # Get the height and width of the slice
    yy, xx = raw_image.shape

    # Create a densities array to store the local densities
    x_pass = int(np.ceil(xx / stride))
    y_pass = int(np.ceil(yy / stride))
    densities = np.zeros((y_pass, x_pass))

    # Tumble over the slice using a fixed window size
    for xi, x in enumerate(np.arange(0, xx, stride)):
        for yi, y in enumerate(np.arange(0, yy, stride)):
            density = np.mean(raw_image[y:y + stride, x:x + stride] < raw_threshold)
            densities[yi, xi] = density
            if density > class_threshold:
                coordinates.append([x, y])
                map_coordinates.append([xi, yi])

    return densities, coordinates, map_coordinates


def return_spectral_image(image):
    """
        Spectral Estimation Image of Stains from Aperio Microscope
        @arg image: NxMx3 loaded image (3d np.array)
        @return: NxM DAB response image
    """

    # These are sensor specific responses for the brain transmission * filter responses, for the AT2 sensor
    # and light-source.  See reports for how to generate them for other sensing systems

    responses = np.array(np.matrix("[ 7.10357511 12.61506218 13.59695489 ; "  # DAB
                                   "8.81961536 10.18302502  5.08567669; "  # Hematoxylin 
                                   "11.02074647 15.41804066 12.77042165 ; "  # Light Source
                                   "17.05035857 17.64819458  9.17788779; "  # Brain Transmission
                                   "0.45574971  4.32897163  1.68161384]"))  # Eosin
    # Convert our image to Nx3 Tristimulus values
    t_i = image.reshape((image.shape[0] * image.shape[1], 3))

    # Least squares fit to our response functions
    fit = np.linalg.lstsq(responses.T, t_i.T, rcond=None)

    # Normalise our responses per map for DAB and Brain Transmission
    DAB = normalise(fit[0][0].copy())
    BT = normalise(fit[0][3].copy())

    # Return DAB - Brain Transmission Function
    raw_DAB = DAB - BT
    raw_DAB = raw_DAB.reshape(image.shape[0], image.shape[1])
    return raw_DAB


def normalise(a):
    """
        Normalise an array, centreing it on 0.5
        @arg a: Array in (np.array NxM)
        @return: transformed values (NxM => 0 < F < 1)
    """
    a += max(np.abs(a.min()), np.abs(a.max()))
    a /= a.max()
    return a


def colourmap_1d(colour1, colour2, one_d_floats):
    """
        Linear map an array of 1d floats between two tristimulus values.
        @arg colour1: 1x3 RGB Value (np.array)
        @arg colour2: 1x3 RGB Value (np.array)
        @arg one_d_floats: array of floats between 0 and 1 (np.array)
        @return: transformed values (NX3 => colour1 < Im < colour2)
    """
    rgbs = np.ones((len(one_d_floats), 3))
    for i in [0, 1, 2]:
        rgbs[:, i] = colour1[i] + one_d_floats * (colour2[i] - colour1[i])
    return rgbs


def colourmap_veridis(one_d_floats):
    """
        Linear map an array of 1d floats between to the veridis colour map
        @arg one_d_floats: array of floats between 0 and 1 (np.array)
        @return: transformed values (NX3 => veridis(0) < Im < veridis(1))
    """
    cmap = matplotlib.cm.get_cmap('viridis')
    return cmap(one_d_floats)[:, 0:3]


def spatial_map_3b3(im):
    """
        Apply a spatial convolutional kernel to im
        @arg im: NxM image (2d or 3d np.array)
        @return: filter image by convolutional kernel
    """
    convolution_kernel = np.array([[0.5 / 8., 0.5 / 8., 0.5 / 8.],
                                   [0.5 / 8., 0.5, 0.5 / 8.],
                                   [0.5 / 8., 0.5 / 8., 0.5 / 8.]])
    result = cv2.filter2D(im, -1, convolution_kernel)
    return result


def hitbox(im_to_hit):
    """
        Draw a hit-box around the edge of an image or image segment
        @arg im_to_hit: array or array slice (3d np.array)
        @return: None, inplace
    """
    im_to_hit[:, 0:10, :] = 0
    im_to_hit[0:10, :, :] = 0
    im_to_hit[-10:-1, :, :] = 0
    im_to_hit[:, -10:-1, :] = 0
    im_to_hit[:, 0:10, 0] = 255
    im_to_hit[0:10, :, 0] = 255
    im_to_hit[-10:-1, :, 0] = 255
    im_to_hit[:, -10:-1, 0] = 255


def hitbox_rgb_image(image, coordinates, stride):
    """
        For a given image and set of coordinates, draw the hitox for those coordinates
        @arg image: array or array slice (3d np.array)
        @arg coordinates: coordinates array referencing y, x location in image
        @arg stride: size of hitbox in px
        @return: image (although it is applied in place)
    """
    for r in coordinates:
        x, y = r
        hitbox(image[y:y + stride, x:x + stride, :])
    return image


def autogain_background(image, xin=(200, -100), yin=(100, 200), threshold=0.95, channel=0, value=255):
    """
        Automatically set pixels around the background mean to maximum white.
        @arg image: NxMx3 loaded image (3d np.array)
        @arg xin: the x coordinates to estimate the background
        @arg yin: the y coordinates to estimate the background
        @arg threshold: the lower bound around the mean for which pixels will be treated as background (default .95/95%)
        @arg channel: the channel to test, by default 0 (RED) as this is not a signal channel
        @arg value: the value to set the R, G and B components to, by default max white (255, 255, 255)
        @return: NxM image with gain adjustment.
    """

    # Test the channel for pixels with value greater than threshold * mean, where mean is the mean of pixels in
    # the region x0,y0 -> x1,y1
    background_mean = np.mean(image[yin[0]:xin[0], yin[1]:xin[1], channel])
    mask_coords = np.asarray(np.column_stack(np.ma.where(image[:, :, channel] > background_mean * threshold)))
    # Set all those pixels to the chosen value
    image[mask_coords[:, 0], mask_coords[:, 1], :] = value

    # return
    return image


def process_file(file, stride, raw_threshold, class_threshold, apply_automatic_background_removal, microns=2.,
                 verbose=False):
    """
        Process an SVS file, loading, decomposing and producing density maps
        @arg file: fullpath to SVS file
        @arg stride: The window size over which we will tumble
        @arg raw_threshold: The value per pixel to be considered a 'vote' for that pixel containing significant signal
        @arg class_threshold: The mean density required for that region to be considered an ROI
        @arg apply_automatic_background_removal: attempt to remove the slide background
        @arg microns: Slide resolution in um per pixel, default 2.0um
        @arg verbose: provide printed output
        @return: image, raw_image (decomposed), densities, coordinates, map_coordinates
    """

    # Load the slide
    rgb_slice = AperioSlide(file) if ".svs" in file else SyntheticSlide(file)

    # Get slide at native resolution
    image = rgb_slice.get_slide_with_pixel_resolution_in_microns(microns=microns)
    if image.shape[0] < stride:
        print("Could not load the SVS file...") if verbose else None
        return None, None, None, None, None

    # Remove the background from the slides if requested.
    if apply_automatic_background_removal:
        print("Removing Slide Background...") if verbose else None
        image = autogain_background(image)

    # This will turn our RGB image into our 6-channel spectral weighted image
    print("Decomposing Slide....") if verbose else None
    raw = return_spectral_image(image)

    # Now we will create density maps and gather the coordinates of significant
    print("Calculating Densities....") if verbose else None
    densities, coordinates, map_coordinates = generate_densities_and_map_coordinates(raw_image=raw,
                                                                                     stride=stride,
                                                                                     raw_threshold=raw_threshold,
                                                                                     class_threshold=class_threshold)
    return image, raw, densities, coordinates, map_coordinates


def create_colourised_density_map(raw, raw_threshold, densities, stride, class_threshold, identify_rois=False):
    """
        Create a colourised, spatially filtered density image, with overlayed map
        @arg raw: raw decomposed image
        @arg raw_threshold: density map threshold
        @arg densities: densities array produced by generate_densities_and_map_coordinates
        @arg stride: stride over which to tumble
        @arg class_threshold : classification threshold for ROIs
        @arg identify_rois: Overlay ROI's of significant density
        @return: reduced_brightness_density_map, density_mapp_array
    """
    # Get the height and width of the slice
    yy, xx = raw.shape

    # Calculate Density on a per pixel bases (for image output)
    density_map_array = (raw < raw_threshold).astype(float)

    # Apply Spatial Filter
    spatial_density_map = spatial_map_3b3(spatial_map_3b3(density_map_array))
    spatial_density_map = colourmap_veridis(spatial_density_map.reshape(raw.shape[0] * raw.shape[1]))
    density_map_array = (spatial_density_map.reshape(raw.shape[0], raw.shape[1], 3) * 255).astype(int)
    reduced_brightness_density_map = density_map_array * 0.5

    # This will produce a spatial gradient in brightness relative to that regions density, atop of the pixel density
    # pseudo-colour
    if identify_rois:
        for xi, x in enumerate(np.arange(0, xx, stride)):
            for yi, y in enumerate(np.arange(0, yy, stride)):
                if densities[yi, xi] < class_threshold:
                    reduced_brightness_density_map[y:y + stride, x:x + stride, :] *= (2 * class_threshold)
                else:
                    reduced_brightness_density_map[y:y + stride, x:x + stride, :] *= (2 * densities[yi, xi])

        density_map_array = reduced_brightness_density_map
    return reduced_brightness_density_map, density_map_array


def write_region_image(parent_path, filename_prefix, coordinates, image, stride):
    """
        Dump ROIs to disk
        @arg parent_path: root folder of images
        @arg filename_prefix: name of the collection of images (e.g. slide name)
        @arg coordinates: coordinates of the start x,y of the regions
        @arg image: image to traverse
        @arg stride: stride over which to tumble
        @return: None
    """
    im_id = 0
    for co in coordinates:
        x, y = co
        seg = image[y:y + stride, x:x + stride]
        io.imwrite(f"{parent_path}/{filename_prefix}_{im_id}.jpg", seg)
        im_id += 1


def process_files_and_folders(input_file_or_folder, output_folder, stride=512, raw_threshold=-0.1, class_threshold=0.1,
                              output_density=False, output_json=False, skip_jpeg=False, auto_remove_background=True,
                              include_only_index=None, output_segmentation=False, verbose=True, synthetic=False):
    """
        Main routine for running this module as a command line function.
        This routine produces density maps, segmentation images and structured json output for a folder or
        single SVS file.

        @arg input_file_or_folder: Full path to SVS file, or folder filled with SVS files
        @arg output_folder: Full path of the folder to output results {json, jpegs}
        @arg stride: Stride over which to tumble
        @arg raw_threshold: Density difference under which we will consider the signal a valid signal
        @arg class_threshold: Mean density identification threshold required to specify an ROI
        @arg output_density: Should density images be saved to disk {jpegs}
        @arg output_json: Should json metadata/density maps be saved to disk {json}
        @arg skip_jpeg: Skip saving the ROI hit-boxed images to disk
        @arg auto_remove_background: Should we apply automatic background identification and removal
        @arg include_only_index: Filter filenames for string, (e.g. "17_A") when not None
        @arg output_segmentation: Output each ROI as an image {jpegs}
        @arg synthetic : Should we look for JPG files

        @arg verbose: printf feedback
        @return: None
    """

    # Handle verbose printing
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    # Make the output directory if it doesn't exist
    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Check input path to see if it is a directory: if so create a list of files to process
    if os.path.isdir(input_file_or_folder):
        list_of_files = []
        for root, directory_names, filenames in os.walk(input_file_or_folder):
            for filename in fnmatch.filter(filenames, '*.svs') if not synthetic else fnmatch.filter(filenames, '*.jpg'):
                if include_only_index is None:
                    list_of_files.append(os.path.join(root, filename))
                else:
                    if include_only_index in filename:
                        list_of_files.append(os.path.join(root, filename))
    else:
        list_of_files = [input_file_or_folder]

    for file_number, file in enumerate(list_of_files):
        # Skip temporary files
        if "._" in file:
            continue
        # Pass the file name to process file for processing and continue
        vprint(f"Processing : {file} | [{file_number + 1} / {len(list_of_files)}]")

        image, raw, densities, coordinates, map_coordinates = process_file(file, stride=stride,
                                                                           raw_threshold=raw_threshold,
                                                                           class_threshold=class_threshold,
                                                                           apply_automatic_background_removal=
                                                                           auto_remove_background)

        if image is None:
            vprint(f"Processing failed for {file}, continuing...")
            continue

        # handle the file and dir case
        parent = os.path.dirname(output_folder)
        name = os.path.basename(file)[:-4]

        if not skip_jpeg:
            vprint("Applying Colourmap to Density Images....")
            reduced_brightness_density_map, density_map = create_colourised_density_map(raw=raw,
                                                                                        raw_threshold=raw_threshold,
                                                                                        densities=densities,
                                                                                        identify_rois=output_density,
                                                                                        stride=stride,
                                                                                        class_threshold=class_threshold)

            # Only classify if we have the minimum number of identified regions
            vprint("Rendering Results....")
            if len(coordinates) > 1:
                image = hitbox_rgb_image(image, coordinates, stride=stride)

            vprint("Writing JPEG....")
            io.imwrite(f"{parent}/density_{name}.jpg", density_map)
            io.imwrite(f"{parent}/{name}.jpg", image)

        if output_segmentation:
            vprint("Saving Segmented Regions....")
            if len(coordinates) > 0:
                write_region_image(parent_path=parent, filename_prefix=name, coordinates=coordinates, image=image,
                                   stride=stride)

        if output_json:
            output_structure = {
                "mapped_coordinates": map_coordinates,
                "image_coordinates": np.array(coordinates).astype(np.int16).tolist(),
                "densities": densities.tolist(),
                "density_threshold_roi": raw_threshold,
                "density_threshold_classification": class_threshold,
                "roi_count": len(coordinates),
                "stride": stride,
            }

            vprint("Writing JSON....")
            with open(f"{parent}/{name}.json", "w") as outfile:
                json.dump(output_structure, outfile)
