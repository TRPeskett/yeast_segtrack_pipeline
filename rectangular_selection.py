""" Script to grab a rectangular selection from image/imagestack
"""

import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector
from PIL import ImageSequence


# Function to be executed after selection
def get_edges(extent, template=False) -> tuple:
    """Get the edges of the selected window

    Args:
        extent (_type_): _description_

    Returns:
        tuple: _description_
    """
    x_min = int(np.floor(extent[0]))
    x_max = int(np.ceil(extent[1]))
    y_min = int(np.floor(extent[2]))
    y_max = int(np.ceil(extent[3]))
    delta_x = int(np.ceil((x_max - x_min)) / 2)
    delta_y = int(np.ceil((y_max - y_min)) / 2)
    if not template:
        return (
            x_min - delta_x,
            x_max + delta_x,
            y_min - delta_y,
            y_max + delta_y,
        )
    return (x_min, x_max, y_min, y_max)


def crop_image(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    input_image_data,
    input_image,
    output_path=Path("./"),
) -> None:
    """Crop an image based on the provided coordinates

    Args:
        x_min (int): x_min
        y_min (int): y_min
        x_max (int): x_max
        y_max (int): y_max
    """
    cropped_ims = ImageSequence.all_frames(
        input_image_data,
        lambda im_frame: im_frame.crop((x_min, y_min, x_max, y_max)),
    )
    output_image = (
        output_path
        / f"{x_min}_{y_min}_{x_max}_{y_max}_{os.path.basename(input_image)}"
    )
    cropped_ims[0].save(
        output_image, save_all=True, append_images=cropped_ims[1:]
    )

    return cropped_ims


def crop_frame(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    input_image_data,
    input_image,
) -> None:
    """Crop an image based on the provided coordinates

    Args:
        x_min (int): x_min
        y_min (int): y_min
        x_max (int): x_max
        y_max (int): y_max
    """
    cropped_ims = ImageSequence.all_frames(
        input_image_data,
        lambda im_frame: im_frame.crop((x_min, y_min, x_max, y_max)),
    )

    return cropped_ims




def extract_data(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    folder,
    input_image,
    mask,
    output_path=Path("./"),
):
    """Extract data corresponding to the cropped image from the hdf5 file

    Args:
        x_min (int): x_min
        y_min (int): y_min
        x_max (int): x_max
        y_max (int): y_max
    """
    subset_rect = (x_min, y_min, x_max, y_max)

    data = h5py.File(mask, "r")
    output_name = (
        output_path
        / f"{x_min}_{y_min}_{x_max}_{y_max}_{os.path.basename(input_image).split('.')[0]}.h5"
    )
    with h5py.File(output_name, "w") as subset:
        for key, _ in data.items():
            # creating the new dataset
            grp = subset.create_group(key)
            # Subsetting the data
            for key2, _ in data[key].items():
                grp.create_dataset(
                    key2,
                    shape=(
                        subset_rect[3] - subset_rect[1],
                        subset_rect[2] - subset_rect[0],
                    ),
                )
                indices = (
                    subset_rect[1],
                    subset_rect[3],
                    subset_rect[0],
                    subset_rect[2],
                )
                grp[key2][:, :] = data[key][key2][
                    indices[0] : indices[1], indices[2] : indices[3]
                ]


def extract_data_from_frame(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    folder,
    input_image,
    mask,
    output_path=Path("./"),
):
    """Extract data corresponding to the cropped image from the hdf5 file

    Args:
        x_min (int): x_min
        y_min (int): y_min
        x_max (int): x_max
        y_max (int): y_max
    """
    subset_rect = (x_min, y_min, x_max, y_max)

    data = h5py.File(mask, "r")
    output_name = (
        output_path
        / f"{x_min}_{y_min}_{x_max}_{y_max}_{os.path.basename(input_image).split('.')[0]}.h5"
    )
    with h5py.File(output_name, "w") as subset:
        for key, _ in data.items():
            # creating the new dataset
            grp = subset.create_group(key)
            # Subsetting the data
            for key2, _ in data[key].items():
                grp.create_dataset(
                    key2,
                    shape=(
                        subset_rect[3] - subset_rect[1],
                        subset_rect[2] - subset_rect[0],
                    ),
                )
                indices = (
                    subset_rect[1],
                    subset_rect[3],
                    subset_rect[0],
                    subset_rect[2],
                )
                grp[key2][:, :] = data[key][key2][
                    indices[0] : indices[1], indices[2] : indices[3]
                ]



def onselect_function(eclick, erelease):
    """Getting the edges of the rectangular selection and cropping the image and data accordingly

    Args:
        eclick (_type_): _description_
        erelease (_type_): _description_
    """
    # Obtain (xmin, xmax, ymin, ymax) values
    # for rectangle selector box using extent attribute.
    # print("Insert custom functionality for the selection here")
    pass


def main() -> None:
    # Zoom the selected part
    # Set xlim range for plot as xmin to xmax
    # of rectangle selector box.
    # plt.xlim(extent[0], extent[1])

    # Set ylim range for plot as ymin to ymax
    # of rectangle selector box.
    # plt.ylim(extent[2], extent[3])
    # return (x_min, x_max, y_min, y_max)

    # plot a line graph for data n
    fig, ax = plt.subplots()

    ax.imshow(img)

    # Define a RectangleSelector at given axes ax.
    # It calls a function named 'onselect_function'
    # when the selection is completed.
    # Rectangular box is drawn to show the selected region.
    # Only left mouse button is allowed for doing selection.
    rect_selector = RectangleSelector(
        ax, onselect_function, drawtype="box", button=[1]
    )

    # Display graph
    plt.show()


if __name__ == "__main__":
    main()
