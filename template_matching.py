"""
Finds instances of a provided or selected template in an image

Written by Tarun Chadha (https://github.com/chadhat). Note that the git history
attributes this file to Nadia Marounina, who committed it; see the Credits
section of the README.
"""

import argparse
import collections
import csv
import logging
import sys
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


# below this fraction of the uint16 range, the fixed 8-bit conversion flattens
# the image enough to cost traps. 2023 movies sit near 0.05, 2021 ones near 0.5.
LOW_RANGE = 0.2


def range_used(image_data):
    """How much of the uint16 range the image actually occupies, 0 to 1."""
    values = np.asarray(image_data, dtype=np.float64)
    return float(values.max() - values.min()) / 65535.0


def should_rescale(image_data, mode='ask', movie_name=''):
    """Whether to rescale before going to 8 bit, warning if the range is small.

    `img_as_ubyte` on raw uint16 divides by a fixed 257, so an image using a
    small part of the range arrives at the matcher nearly flat. The 2023 ageing
    movies span about 5% of the range and reach it as values 2 to 19 out of 255;
    the 2021 movies span about 50% and are fine. Measured on `2023-07-25 Pos22`
    against a template cut from Pos26, rescaling took the peak match from 0.79 to
    0.91 and the traps found from 39 to 50, where 53 is what the position's own
    template finds. On the 2021 movies it changed nothing at all.

    mode : 'ask' warns and prompts, 'always' and 'never' decide without asking.
           A run with nothing attached to its input cannot be asked, so it
           rescales and says so rather than hanging on a prompt nobody will see.
    """
    if mode == 'always':
        return True
    if mode == 'never':
        return False

    fraction = range_used(image_data)
    if fraction >= LOW_RANGE:
        return False

    print()
    print(f"  WARNING: {movie_name or 'this movie'} uses only "
          f"{100 * fraction:.1f}% of the 16-bit range.")
    print("  Converting it to 8 bit as-is leaves the trap detection almost no")
    print("  contrast to work on, and traps get missed - about half of them on")
    print("  the movie this was measured on. Rescaling the image to its own")
    print("  range first fixes that, and changes nothing on a movie that already")
    print("  uses the range well.")

    if not sys.stdin or not sys.stdin.isatty():
        print("  Not running interactively, so rescaling. Pass --rescale never "
              "to keep the old behaviour.")
        return True

    answer = input("  Rescale before detecting traps? [Y/n] ").strip().lower()
    return answer in ('', 'y', 'yes')


def preprocess(image_data, rescale=False):
    """Convert an input image to 8 bit, optionally rescaling it first.

    rescale : stretch the image to its own 0.1-99.9 percentile before converting,
              instead of dividing the raw uint16 by a fixed 257. The percentiles
              rather than min and max so that one hot pixel cannot set the scale.
    """
    logging.info("Converting the image to 8bit")

    values = np.array(image_data, dtype=np.uint16)
    if not rescale:
        return skimage.util.img_as_ubyte(values)

    low, high = np.percentile(values.astype(np.float64), (0.1, 99.9))
    stretched = (values.astype(np.float64) - low) / max(high - low, 1.0)
    return skimage.util.img_as_ubyte(np.clip(stretched, 0.0, 1.0))


def template_matching(
        image_data,
        template,
        output_folder,
        threshold=0.6,
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

def single_trap_split(input_image, input_mask, output_folder):
    """Treat the whole movie as one trap, skipping the search for traps in it.

    The annotated ground-truth data is supplied as crops of individual traps that
    were made by hand, so there is nothing left to find: matching a template
    against a crop that is already one trap would at best re-crop it tighter and
    lose the margin the later steps need, and at worst find nothing at all.

    Writes the same layout the template-matching path writes - one numbered
    folder under split_data holding the movie and its mask - so everything
    downstream is unchanged.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = Path(output_folder) / timestamp
    output_folder.mkdir(parents=True, exist_ok=True)

    raw_image_data, first_frame = read_image(input_image)
    width, height = first_frame.size

    logging.info(f"Treating {input_image} as a single trap of {width}x{height}")

    split_data(input_image,
               raw_image_data,
               {0: (0, 0, width, height)},
               input_mask,
               output_folder)

    return output_folder


def main(input_image,
         input_mask,
         template_image,
         output_folder,
         threshold=0.7,
         extra_width=0.9,
         extra_height=0.9,
         methods=('TM_CCOEFF_NORMED',),
         match_image=None,
         rescale='ask'
         ):
    """Template matching pipeline

    match_image : image to search for the traps in, if not the first frame of the
                  movie. The cell-free background is a much better thing to match
                  against: a generated template is a median over traps and so has
                  the noise of no particular frame, and matching it against one
                  raw frame scores ~0.69 where matching it against the background
                  scores ~0.96. It also means cells can no longer sit on a trap
                  and stop it being found.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # a template given as a bare name, or not given at all, has no parent
    # directory to read a previous run's timestamp out of
    parts = str(template_image).split('/')
    potential_timestamp = parts[-2] if len(parts) > 1 else ''

    if not os.path.isdir(Path(output_folder) / timestamp) and not os.path.isdir(
            Path(output_folder) / potential_timestamp):

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

        to_match = raw_image_data if match_image is None else match_image
        # decided once, from the image the traps are actually looked for in, and
        # then applied to the template too - the two have to reach the matcher
        # on the same scale or the scores mean nothing
        rescaling = should_rescale(to_match, rescale, str(input_image))
        image_array = preprocess(to_match, rescaling)
        template_array = preprocess(template_data, rescaling)

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

        output_folder = Path(output_folder) / potential_timestamp

    return output_folder

# if __name__ == "__main__":
#     main()
