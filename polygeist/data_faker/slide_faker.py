"""
Functions that produce fake slide data.
"""
from perlin_noise import PerlinNoise
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np

# Default config (for reference and default)
_default_config = {
    # Random seed to generate a slide
    "seed": 1,
    # Size of each patch to generate
    "window_size": (256, 256),
    # Total slide size
    "slide_size": (256 * 12, 256 * 12),
    # Slide Luminance
    "luminance": 240,
    # Background slide colour (non stained cellular matter)
    "background_rgb": (200, 210, 219),
    # Non-stained nuclei
    "cell_nucleus_rgb": (171, 176, 255),
    # Non-stained bodies similar to stained asyn
    "melanin_rgb": (74, 67, 53),
    # Stained asyn
    "a-syn_rgb": (125, 94, 15),
    # The probability of Asyn being present in each window
    # (set to a smaller number for controls)
    "probability_of_a-syn": 0.15,
}
"""Configuration structure that is used to generate pseudo-slides"""


def _clip_upper_and_lower(value, lower, upper):
    return min(upper, max(value, lower))


# Based on @MitchWheat, @templatetypedef & @Ounsworth algorithm
def _random_angular_steps(steps, irregularity):
    lower = float((2 * np.pi / steps) - irregularity)
    upper = float((2 * np.pi / steps) + irregularity)
    _range = np.arange(steps)
    angles = [np.random.uniform(lower, upper) for _ in _range]
    cumulative_sum = np.sum(angles) / (2.0 * np.pi)
    n_angles = [angle / cumulative_sum for angle in angles]
    return n_angles


def _generate_cell_mask(size, centroid, avg_radius, vertex_noise, peaks, num_vertices):
    # Parameter check
    if vertex_noise < 0 or vertex_noise > 1 or peaks < 0 or peaks > 1:
        raise ValueError("vertex_noise and peaks must between 0 - 1")

    vertex_noise *= 2 * np.pi / num_vertices
    peaks *= avg_radius
    angle_steps = _random_angular_steps(num_vertices, vertex_noise)

    # now generate the points
    angle = np.random.uniform(0, 2 * np.pi)
    angles = [angle]
    for a in angle_steps:
        angle += a
        angles.append(angle)

    radii = [
        _clip_upper_and_lower(
            np.random.normal(avg_radius, peaks), 0.0, 2.0 * avg_radius
        )
        for _ in angles
    ]
    points = [
        (centroid[0] + radius * np.cos(angle), centroid[1] + radius * np.sin(angle))
        for radius, angle in zip(radii, angles)
    ]

    black = (0, 0, 0)
    white = (255, 255, 255)
    img = Image.new("RGB", size, black)
    draw = ImageDraw.Draw(img)
    draw.polygon(points, outline=black, fill=white)

    return img


def generate_fake_slide_data(
    seed=1,
    xpix_ypix=(256, 256),
    sx_sy=(256 * 12, 256 * 12),
    b_rgb=(200, 210, 219),
    c_rgb=(171, 176, 255),
    m_rgb=(74, 67, 53),
    a_rgb=(125, 94, 15),
    a_prob=0.15,
    Y=240,
):
    """
    Produces an image file with the specified features.
    @arg seed: Psudorandom seed
    @arg xpix_ypix: Size of each patch to generate (window_size)
    @arg sx_sy: Total slide size (slide_size)
    @arg b_rgb: Background slide colour (non stained cellular matter)
    @arg c_rgb: Non-stained nuclei (cell_nucleus_rgb)
    @arg m_rgb: Non-stained bodies similar to stained asyn (melanin_rgb)
    @arg a_rgb: Stained asyn (a-syn_rgb)
    @arg a_prob: Probability of Asyn being present in each window
    @arg Y: Slide Luminance
    @return: Image (np.array)
    """
    # Unpack tuples
    xpix, ypix = xpix_ypix
    sx, sy = sx_sy

    # Seed for numpy
    np.random.seed(seed=seed)

    # Generate our slide
    slide = np.zeros((sy, sx, 3))
    # Create a mask for the actual slide region
    _mask = _generate_cell_mask(
        size=(sy, sx),
        centroid=(sy / 2, sx / 2),
        avg_radius=sy / 2.5,
        vertex_noise=0.35,
        peaks=0.2,
        num_vertices=16,
    )
    # convert to numpy so we can use it as a mask
    mask = np.array(_mask.getdata()).reshape(_mask.size[0], _mask.size[1], 3)

    # iteration counter for the seed
    iteration = 0
    for y in tqdm(np.arange(0, sy - ypix, ypix)):
        for x in np.arange(0, sx - xpix, xpix):

            # Generate some perlin noise
            noise = PerlinNoise(octaves=10, seed=seed + iteration)

            # Create an array for our segment
            segment = np.zeros((ypix, xpix, 3), dtype=np.uint8)

            # Background, a light blue stain
            segment[:, :, 0] = b_rgb[0]
            segment[:, :, 1] = b_rgb[1]
            segment[:, :, 2] = b_rgb[2]

            # Create some cell bodies
            x_ = (np.random.rand(xpix) * xpix).astype(int)
            y_ = (np.random.rand(ypix) * ypix).astype(int)

            # Cell Nuclei
            segment[y_, x_, 0] = c_rgb[0]
            segment[y_, x_, 1] = c_rgb[1]
            segment[y_, x_, 2] = c_rgb[2]

            # Generate some perlin noise
            pic = np.array(
                [
                    [noise([i / xpix, j / ypix]) for j in range(xpix)]
                    for i in range(ypix)
                ]
            )

            # Create fake melanin/cell bodies
            binary_mask = (pic > 0.2) & (pic < 0.25)
            segment[binary_mask, 0] = m_rgb[0]
            segment[binary_mask, 1] = m_rgb[1]
            segment[binary_mask, 2] = m_rgb[2]

            if np.random.random() > (1.0 - a_prob):
                # Create fake a-syn bodies
                binary_mask = (pic > 0.45) & (pic < 0.6)
                segment[binary_mask, 0] = a_rgb[0]
                segment[binary_mask, 1] = a_rgb[1]
                segment[binary_mask, 2] = a_rgb[2]

            # Regenerate some new noise to create overlapping striped bodies
            noise = PerlinNoise(octaves=10, seed=seed + 1024 + iteration)
            pic = np.array(
                [
                    [noise([i / xpix, j / ypix]) for j in range(xpix)]
                    for i in range(ypix)
                ]
            )

            # Should region contain asyn?
            if np.random.random() > (1.0 - a_prob):
                # Extra cellular a-syn
                binary_mask = (pic > 0.004) & (pic < 0.008)
                segment[binary_mask, 0] = a_rgb[0]
                segment[binary_mask, 1] = a_rgb[1]
                segment[binary_mask, 2] = a_rgb[2]

                # Create more noise to create extra cellular bodies
                noise = PerlinNoise(octaves=10, seed=seed + 2048 + iteration)
                pic = np.array(
                    [
                        [noise([i / xpix, j / ypix]) for j in range(xpix)]
                        for i in range(ypix)
                    ]
                )

                # Create fake a-syn bodies outside cells
                binary_mask = (pic > 0.4) & (pic < 0.6)
                segment[binary_mask, 0] = a_rgb[0]
                segment[binary_mask, 1] = a_rgb[1]
                segment[binary_mask, 2] = a_rgb[2]

            # Use the last generated field to create luminance deviation.
            deviation = 20 - (pic * 30).astype(np.uint8)
            segment[:, :, 0] += deviation
            segment[:, :, 1] += deviation
            segment[:, :, 2] += deviation

            slide[y : y + ypix, x : x + xpix, :] = segment.copy()
            iteration += 1

    # Simulate cortex area, with white slide background
    slide[slide < 1] = Y
    slide[~(mask[:, :, 0] > 0), :] = Y
    return slide
