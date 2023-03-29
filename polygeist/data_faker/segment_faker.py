import numpy as np
import imageio as io


def generate_fake_segment(x_size=100, y_size=100, dab_contrib=1.0):
    # Leica responses
    responses = np.array(
        [
            [7.10357511, 12.61506218, 13.59695489],
            [8.81961536, 10.18302502, 5.08567669],
            [11.02074647, 15.41804066, 12.77042165],
            [17.05035857, 17.64819458, 9.17788779],
            [0.45574971, 4.32897163, 1.68161384],
        ]
    )

    # The amount of dab to inject is controlled by the input scalar
    responses[3, :] *= dab_contrib

    # generate the size as requested
    image = np.zeros((y_size, x_size, 3))
    image_crosstalk = np.zeros((y_size, x_size, 3))

    # Put indices in the first element
    image[:, :, 0] = (np.random.random((y_size, x_size)) * 4).astype(int)
    image_crosstalk[:, :, 0] = (np.random.random((y_size, x_size)) * 4).astype(int)

    # replace elements with pure signal
    for i in np.arange(0, 5):
        image[image[:, :, 0] == i, :] = responses[i, :] * 0.9
        image_crosstalk[image[:, :, 0] == i, :] = responses[i, :] * 0.1

    # Create a scalar field
    mulfield = np.random.random((y_size, x_size)) * 10

    # Apply the field
    scalar = np.zeros((y_size, x_size, 3))
    for i in np.arange(0, 3):
        scalar[:, :, i] = mulfield

    # Create the final image
    image = np.multiply(image, scalar)
    image_crosstalk = np.multiply(image_crosstalk, scalar)

    image = image + image_crosstalk

    return image.astype(np.uint8)


def write_fake_data_segments(
    directory,
    pathology_present=False,
    case_name="PDXXX",
    slide_index=17,
    segments=10,
    pathology="A-syn",
    stride=512,
):
    for i in np.arange(0, segments):
        im = generate_fake_segment(
            x_size=stride, y_size=stride, dab_contrib=1.0 if pathology_present else 0.25
        )
        io.imwrite(f"{directory}/{case_name}-{slide_index}_{pathology}{i}.jpg", im)
