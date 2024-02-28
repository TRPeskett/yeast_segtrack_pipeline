"""
Finds instances of a provided or selected template in an image
"""

import argparse
import collections
import csv
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import skimage
from cv2 import TM_CCOEFF_NORMED, circle, imwrite, matchTemplate, rectangle
from PIL import Image, ImageSequence
from rectangular_selection import *


def create_summary(input_file, time_value1=0, time_value2=0):
    """Create a list of unique cells present at each time step

    Args:
        input_file (_type_): _description_
    """
    with h5py.File(input_file.replace(".tif", ".h5"), "r") as h5file:
        data = h5file["FOV0"]
        # data = collections.OrderedDict(sorted(data.items()))
        summary = {}
        for key, value in data.items():
            summary[key] = list(np.unique(value[:, :]))
            summary[key].remove(0.0)
        for key in range(time_value1, time_value2 + 1):
            summary[key] = summary.pop(f"T{key}")
        summary = collections.OrderedDict(sorted(summary.items()))
    with open(
            input_file.replace(".tif", "csv"), "w", encoding="utf-8"
    ) as csv_file:
        writer = csv.writer(csv_file)
        for key, value in summary.items():
            writer.writerow([key, value])


def range_limited(MIN_VAL=0.0, MAX_VAL=1.0):
    """Return function handle of an argument type function for
    ArgumentParser checking a float range: MIN_VAL <= arg <= MAX_VAL
      MIN_VAL - minimum acceptable argument
      MAX_VAL - maximum acceptable argument"""

    # Define the function with default arguments
    def range_limited_float_type(arg):
        """Type function for argparse - a float within some predefined bounds"""
        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("Must be a floating point number")
        if f < MIN_VAL or f > MAX_VAL:
            raise argparse.ArgumentTypeError(
                "Argument must be < " + str(MAX_VAL) + "and > " + str(MIN_VAL)
            )
        return f

    return range_limited_float_type


def arguments():
    """Parsing the input arguments"""

    parser = argparse.ArgumentParser(
        description="Help for seperating the traps and running re-tracking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input_image",
        help="Input image including the full path for extraction",
        type=str,
        default="/Users/tarunchadha/Documents/yeast_ageing/ageing_movie_examples/2021-12-15_ageing_optoControls_lightOff_Pos9_BF-1.tif",
        # required=True,
    )
    parser.add_argument(
        "-t",
        "--template_image",
        help="Template image including the full path for extraction",
        type=str,
        # default="/media/chadhat/4d105adc-3356-4a16-9761-ee0dcd7f23dc/Work/yeast_ageing/template_matching/data/trap_template.png",
    )
    parser.add_argument(
        "-im",
        "--input_mask",
        help="Input mask including the full path for extraction",
        type=str,
        default="/Users/tarunchadha/Documents/yeast_ageing/ageing_movie_examples/mask_clean_data_batchsize_2_Nepochs_250_21_04.h5",
    )
    parser.add_argument(
        "-th",
        "--threshold",
        help="Threshold for template matching",
        type=range_limited(MIN_VAL=0.0, MAX_VAL=1.0),
        default=0.7,
    )
    parser.add_argument(
        "-m",
        "--methods",
        help=(
            "methods for template matching. Some of the possible values are"
            " TM_CCOEFF_NORMED, TM_CCORR_NORMED, TM_SQDIFF_NORMED"
        ),
        type=str,
        default="TM_CCOEFF_NORMED",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        help="Folder where the output files will be saved",
        type=str,
        default="./output/",
    )
    parser.add_argument(
        "-extra_width",
        help=(
            "Width on either side of the detected trap to be included in the"
            " output"
        ),
        type=range_limited(MIN_VAL=0, MAX_VAL=1.5),
        default=0.9,
    )
    parser.add_argument(
        "-extra_height",
        help=(
            "Height on either side of the detected trap to be included in the"
            " output"
        ),
        type=range_limited(MIN_VAL=0, MAX_VAL=1.5),
        default=0.9,
    )
    return parser.parse_args()


def read_image(input_image, frame_nb=0):
    """Read the input image/image stack and return the first image

    Args:
        input_image (Path): input image including the full path

    Returns:
        numpy array: Array containing the input image or the first image of an
        input image stack
    """

    logging.info(f"Reading image {input_image}")

    data = Image.open(input_image)

    frames = ImageSequence.all_frames(data)
    # converted_image = np.array(frame, dtype=np.uint16)
    first_frame = frames[frame_nb]

    """
    
    converted_image = skimage.io.imread(input_image)
    if len(converted_image.shape)==3:
        print(type(converted_image[0,:,:]))
        return converted_image[0,:,:]
    """

    return data, first_frame


def preprocess(image_data):
    """Converts an input image to 8bit if it is not already

    Args:
        image (nd.array): _description_

    Returns:
        nd.array: _description_
    """

    logging.info("Converting the image to 8bit")

    return skimage.util.img_as_ubyte(np.array(image_data, dtype=np.uint16))


def template_matching(
        image_data,
        template,
        output_folder,
        threshold=0.7,
        width_height=(0.9, 0.9),
        methods=("TM_CCOEFF_NORMED",),
):
    """Function to find instances of a template in an image

    Args:
        image (nd.array): input image
    """

    logging.info(
        "Finding instances of the provided or selected template in the image"
    )

    # img_rgb = cv.imread(image)
    # img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    # template = cv.imread("mario_coin.png", 0)
    width, height = template.shape[::-1]
    width_im, height_im = image_data.shape[::-1]
    crop_points = {}
    for method in methods:
        print(eval(method))
        res = matchTemplate(image_data, template, eval(method))
        loc = np.where(res >= threshold)
        image_data_tmp = image_data
        mask = np.zeros(image_data.shape[:2], np.uint8)
        index = 0
        for _, point in enumerate(zip(*loc[::-1])):
            # print(pt)
            # print(pt[0] + w, pt[1] + h)
            # cv.rectangle(image_data_tmp, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
            if (
                    mask[
                        point[1] + int(round(height / 2)),
                        point[0] + int(round(width / 2)),
                    ]
                    != 255
            ):
                mask[
                point[1]: point[1] + height, point[0]: point[0] + width
                ] = 255
                rectangle(
                    image_data_tmp,
                    point,
                    (point[0] + width, point[1] + height),
                    (0, 0, 255),
                    2,
                )
                """
                circle(
                    image_data_tmp,
                    (point[0], point[1]),
                    radius=1,
                    color=(0, 0, 255),
                    thickness=1,
                )
                """
                (x_min, x_max, y_min, y_max) = (
                    max(0, int(point[0] - width / 2)),
                    min(width_im, int(point[0] + 3 * width / 2)),
                    max(0, int(point[1] - height / 2)),
                    min(height_im, int(point[1] + 3 * height / 2)),
                )
                # rectangle(
                #    image_data_tmp, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2
                # )
                crop_points[index] = (x_min, y_min, x_max, y_max)
                index += 1
        imwrite(
            f"{output_folder}/detected_traps_using_{method}.png",
            image_data_tmp,
        )
    return crop_points


def split_data(input_image, raw_image_data, crop_points, input_mask, output_folder):
    logging.info("Splitting the image and data based on the detected traps")
    try:
        os.mkdir(
            Path(output_folder) / "split_data",
        )
    except FileExistsError:
        pass
    for key, points in crop_points.items():
        output_path = Path(output_folder) / "split_data" / f"{key}"
        try:
            os.mkdir(output_path)
        except FileExistsError:
            pass
        crop_image(
            *points,
            input_image_data=raw_image_data,
            input_image=input_image,
            output_path=output_path,
        )
        extract_data(
            *points,
            key,
            input_image=input_image,
            mask=input_mask,
            output_path=output_path,
        )


def check_args(args):
    """Check if the provided arguments are valid"""

    logging.info("Checking the arguments provided by the user")
    if args.input_image:
        if not os.path.isfile(args.input_image):
            raise FileNotFoundError(
                f"Input image {args.input_image} not found"
            )
    if args.template_image:
        if not os.path.isfile(args.template_image):
            raise FileNotFoundError(
                f"Template image {args.template_image} not found"
            )

    if args.input_mask:
        if not os.path.isfile(args.input_mask):
            raise FileNotFoundError(f"Input mask {args.input_mask} not found")

    if args.output_folder:
        if not os.path.isdir(args.output_folder):
            raise FileNotFoundError(
                f"Output folder {args.output_folder} not found"
            )


def create_template(img, input_image, output_folder):
    """Get the template from the image using a rectangular selection"""

    logging.info("Creating template based on user selection")

    _, ax = plt.subplots()

    ax.imshow(img)

    # Define a RectangleSelector at given axes ax.
    # It calls a function named 'onselect_function'
    # when the selection is completed.
    # Rectangular box is drawn to show the selected region.
    # Only left mouse button is allowed for doing selection.
    rect_selector = RectangleSelector(ax, onselect_function, button=[1])

    # Display graph
    plt.show()
    extent = rect_selector.extents
    # print("Extents: ", extent)
    x_min, x_max, y_min, y_max = get_edges(extent, template=True)
    # img_crop = img.crop((x_min,y_min, x_max, y_max))
    template = crop_image(
        x_min,
        y_min,
        x_max,
        y_max,
        input_image_data=img,
        input_image=input_image,
        output_path=Path(output_folder),
    )
    # extract_data(
    #    x_min, y_min, x_max, y_max, input_image=input_image, mask=input_mask
    # )
    return template


# def main() -> None:
#     """Template matching pipeline"""
#
#     timestamp = datetime.now().strftime("%Y-%m-%d")
#
#     args = arguments()
#
#     # check if the provided arguments are valid
#     check_args(args)
#
#     if not os.path.isdir(Path(args.output_folder) / timestamp):
#         os.mkdir(Path(args.output_folder) / timestamp)
#         args.output_folder = Path(args.output_folder) / timestamp
#
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s %(levelname)s %(message)s",
#         filename=f"{args.output_folder}/template_matching.log",  # the log file name
#         filemode="w",  # mode in which to open the file (write mode here)
#     )
#
#     raw_image_data, first_image_data = read_image(args.input_image)
#
#     if args.template_image:
#         template_data, _ = read_image(args.template_image)
#     else:
#         [template_data] = create_template(first_image_data, args)
#
#     image_array = preprocess(raw_image_data)
#
#     template_array = preprocess(template_data)
#     print("Args: ", args)
#     crop_points = template_matching(
#         image_array,
#         template_array,
#         args.output_folder,
#         threshold=args.threshold,
#         width_height=(args.extra_width, args.extra_height),
#         methods=tuple(args.methods.split(",")),
#     )
#
#     split_data(args.input_image, raw_image_data, crop_points, args)

def main(input_image,
         input_mask,
         template_image,
         output_folder,
         threshold=0.7,
         extra_width=0.9,
         extra_height=0.9,
         methods=('TM_CCOEFF_NORMED',)
         ):
    """Template matching pipeline"""

    timestamp = datetime.now().strftime("%Y-%m-%d")

    if not os.path.isdir(Path(output_folder) / timestamp):
        os.mkdir(Path(output_folder) / timestamp)
        output_folder = Path(output_folder) / timestamp

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            filename=f"{output_folder}/template_matching.log",  # the log file name
            filemode="w",  # mode in which to open the file (write mode here)
        )

        raw_image_data, first_image_data = read_image(input_image)

        if template_image:
            template_data, _ = read_image(template_image)
        else:
            [template_data] = create_template(first_image_data,
                                              input_image,
                                              output_folder)

        image_array = preprocess(raw_image_data)
        template_array = preprocess(template_data)

        crop_points = template_matching(
            image_array,
            template_array,
            output_folder,
            threshold=threshold,
            width_height=(extra_width, extra_height),
            methods=methods,
        )

        split_data(input_image,
                   raw_image_data,
                   crop_points,
                   input_mask,
                   output_folder)
    else:

        output_folder = Path(output_folder) / timestamp

    return output_folder

# if __name__ == "__main__":
#     main()
