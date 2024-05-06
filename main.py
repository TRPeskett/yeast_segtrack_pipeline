import template_matching

import tqdm
from dotenv import load_dotenv

from yeaz.unet.segment import segment
from yeaz.disk import Reader as nd
import argparse
import skimage
from yeaz.unet import neural_network as nn

import torch
import glob
import os
import h5py
from tiffwrite import tiffwrite
from pathlib import Path
from skimage import io
from PIL import Image, ImageSequence

from midap.tracking.bayesian_tracking import BayesianCellTracking

import numpy as np
from skimage.registration import phase_cross_correlation
from scipy import ndimage
import posttreatment
import fluorescence


def LaunchPrediction(im, mic_type, pretrained_weights=None, device='cpu'):
    """It launches the neural neutwork on the current image and creates
    a hdf file with the prediction for the time T and corresponding FOV.
    """
    im = skimage.exposure.equalize_adapthist(im)
    im = im * 1.0
    pred = nn.prediction(im, mic_type, pretrained_weights, device=device)
    return pred


def ThresholdPred(thvalue, pred):
    """Thresholds prediction with value"""
    if thvalue is None:
        thresholdedmask = nn.threshold(pred)
    else:
        thresholdedmask = nn.threshold(pred, thvalue)
    return thresholdedmask


def LaunchInstanceSegmentation(reader, image_type,
                               fov_indices=None, time_value1=0, time_value2=0,
                               thr_val=None,
                               min_seed_dist=5,
                               path_to_weights=None, device='cpu'):
    if fov_indices is None:
        fov_indices = [0]
    if (device == 'cuda') and torch.cuda.is_available():
        f_device = 'cuda'
    else:
        f_device = 'cpu'

        # cannot have both path_to_weights and image_type supplied
    if (image_type is not None) and (path_to_weights is not None):
        print("image_type and path_to_weights cannot be both supplied.")
        return

    # check if correct imaging value
    if (image_type not in ['bf', 'pc']) and (path_to_weights is None):
        print("Wrong imaging type value ('{}')!".format(image_type),
              "imaging type must be either 'bf' or 'pc'")
        return

    # check range_of_frames constraint
    if time_value1 > time_value2:
        print("Error", 'Invalid Time Constraints')
        return

    # displays that the neural network is running
    print('Running the neural network on {} ...'.format(f_device))

    for fov_ind in tqdm.tqdm(fov_indices, desc='FOV', position=0):

        # iterates over the time indices in the range
        for t in tqdm.tqdm(range(time_value1, time_value2 + 1), desc='Time', position=1, leave=False):
            # print('--------- Segmenting field of view:',fov_ind,'Time point:',t)

            # calls the neural network for time t and selected fov
            im = reader.LoadOneImage(t, fov_ind)

            try:
                pred = LaunchPrediction(im, image_type, pretrained_weights=path_to_weights, device=f_device)
            except ValueError:
                print('Error! ',
                      'The neural network weight files could not '
                      'be found. \nMake sure to download them from '
                      'the link in the readme and put them into '
                      'the folder unet, or specify a path to a custom weights file with -w argument.')
                return

            thresh = ThresholdPred(thr_val, pred)
            seg = segment(thresh, pred, min_seed_dist)
            reader.SaveMask(t, fov_ind, seg)
            print('--------- Finished segmenting.')

            # apply tracker if wanted and if not at first time
            temp_mask = reader.CellCorrespondence(t, fov_ind)
            reader.SaveMask(t, fov_ind, temp_mask)


def correct_shift_in_mask(mask: str, list_shifts: dict) -> None:
    '''
    Correct the drift/shift of the movie on maskfiles.
    Will *replace* the original mask with the corrected mask

    mask : path to mask file to correct
    list_shifts : dictionary containing the shifts per frame
    '''
    with h5py.File(mask, "r+") as f_masks:
        for group in f_masks.keys():
            for dset in f_masks[group].keys():
                ds_data = f_masks[group][dset][:]

                frame_nb = int(dset.split('T')[-1])

                shifted_mask = ndimage.shift(ds_data, list_shifts[frame_nb])
                f_masks[group][dset][...] = shifted_mask


def correct_shift(path_to_image: str, mask: str) -> (list, dict):
    '''
    Will correct the shift/drift of the movie (stack tif file) with respect to the
    first frame. The corrected images are the output of the function.
    The corresponding masks will also be corrected by calling
    the correct_shift_in_mask routine.

    path_to_image : path to the image that needs shift correction
    mask : path to mask corresponding to the image
    '''
    data = Image.open(path_to_image)
    frames = ImageSequence.all_frames(data)

    first_frame = skimage.util.img_as_ubyte(np.array(frames[0], dtype=np.uint16))

    list_of_shifted_images = []
    list_shifts = {}
    for i in range(0, len(frames)):
        current_frame = skimage.util.img_as_ubyte(np.array(frames[i], dtype=np.uint16))
        shift, error, diffphase = phase_cross_correlation(first_frame, current_frame)
        frame_shifted = ndimage.shift(frames[i], shift)
        frame_shifted = frame_shifted.astype(np.uint16)
        one_image = Image.fromarray(frame_shifted)

        list_of_shifted_images.append(one_image)
        list_shifts[i] = shift

    correct_shift_in_mask(mask, list_shifts)

    return list_of_shifted_images, list_shifts


def adapt_output_to_midap(root_path: str, frame_ini: int, frame_end: int) -> None:
    '''
    This routine will take in a stack of tif images and the corresponding h5 mask
    and then will create the midap folder, containing :
    - cut_im = one png image per frame, named to respec the convention expected in midap
    - seg_im = masks in tif format, one per frame, same expected naming convention

    root_path: path to the outputs
    frame_ini: initial frame (usually 0)
    frame_end: final frame of the movie
    '''

    list_traps = glob.glob(str(root_path) + '/split_data/*')
    list_traps.sort()

    for i, trap_path in enumerate(list_traps):
        print(f"Converting to midap format trap {trap_path:s}")

        maskfile = glob.glob(list_traps[i] + '/*h5')[0]
        tiffile = glob.glob(list_traps[i] + '/*tif')[0]

        list_shifted_im, dict_shifts = correct_shift(tiffile, maskfile)
        list_shifted_im[0].save(list_traps[i] + '/shift_corrected.tif',
                                save_all=True,
                                append_images=list_shifted_im[1:])

        tiffile = list_traps[i] + '/shift_corrected.tif'

        # with open(list_traps[i] + '/saved_shift.pkl', 'wb') as f:
        #     pickle.dump(dict_shifts, f)

        Path(trap_path + '/midap/seg_im').mkdir(parents=True, exist_ok=True)
        Path(trap_path + '/midap/cut_im').mkdir(parents=True, exist_ok=True)

        mask = h5py.File(maskfile, "r")
        group = "FOV0"
        list_dset_num = range(frame_ini, frame_end + 1)

        for n in list_dset_num:
            dset = 'T' + str(n)
            arr = mask[group][dset][:]
            arr = arr.astype(np.uint16)

            tiffwrite(trap_path + '/midap/seg_im/im_frame' + f"{n:03d}" + '_seg.tif', arr, 'XY')

        im = io.imread(tiffile)
        for n in range(frame_ini, frame_end + 1):
            im_for_png = Image.fromarray(im[n])
            im_for_png.save(trap_path + "/midap/cut_im/im_frame" + f"{n:03d}" + "_cut.png")

def arguments():
    """Parsing the input arguments"""

    parser = argparse.ArgumentParser(
        description="Run the segmentation and tracking pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input_movie",
        help="Path to the input movie",
        type=str,
        required=True
    )

    parser.add_argument(
        "-fo",
        "--fluor_offset",
        help="How many frames before the first fluorescence frame ?",
        type=int,
        required=True
    )

    parser.add_argument(
        "-fs",
        "--fluor_step",
        help="How many frames in between fluorescence frames ?",
        type=int,
        required=True
    )

    parser.add_argument(
        "-tx",
        "--trap_x",
        help="width of the center of the trap, in px",
        type=int,
        default=-1,
        required=True
    )

    parser.add_argument(
        "-ty",
        "--trap_y",
        help="height of the center of the trap, in px",
        type=int,
        default=-1,
        required = True
    )

    parser.add_argument(
        "-t",
        "--template",
        help="Path to a template of an empty trap",
        type=str,
        default=''
    )

    parser.add_argument(
        "-no_s",
        "--no_segmentation",
        help="Exclude the segmentation step from the pipeline",
        default=False,
        action='store_true'
    )

    parser.add_argument(
        "-no_fm",
        "--no_format_midap",
        help="Exclude the formatting of the data for midap",
        default=False,
        action='store_true'
    )

    parser.add_argument(
        "-no_tr",
        "--no_tracking",
        help="Exclude the tracking step",
        default=False,
        action='store_true'
    )

    parser.add_argument(
        "-no_m",
        "--no_mother",
        help="Exclude the detection of the mother cell",
        default=False,
        action='store_true'
    )

    parser.add_argument(
        "-no_fl",
        "--no_fluor",
        help="Exclude the treatment of the fluoresence data",
        default=False,
        action='store_true'
    )

    return parser.parse_args()


def main():
    load_dotenv()
    args = arguments()

    path_to_full_movie = args.input_movie
    path_to_weights = os.getenv('WEIGHTS_YEAST')
    path_to_template = args.template

    path_to_first_mask = path_to_full_movie.split(".tif")[0]
    path_to_first_mask = path_to_first_mask + '.h5'
    filename = path_to_full_movie.split('/')[-1]
    filename = filename.split('.')[0]

    fluor_step = args.fluor_step
    fluor_offset = args.fluor_offset

    im = io.imread(path_to_full_movie)
    Nmax, Ndim1, Ndim2 = im.shape
    frame_ini = 0
    frame_end = Nmax - 1

    trap_center_size = [args.trap_y, args.trap_x]
    folder_init_movie = Path(path_to_full_movie)
    folder_init_movie = str(folder_init_movie.parent)

    ###################################
    # step 1 : do the segmentation of the full movie [code from launch_NN_commandline]
    # Including the fluor frames. Ignore fluor segmentation and separate the fluor
    # from non-fluor frames in a post-processing step
    ###################################

    if not args.no_segmentation:
        print('\n')
        print('Running the segmentation :')
        reader = nd.Reader("", path_to_first_mask, path_to_full_movie)
        LaunchInstanceSegmentation(reader, image_type=None, fov_indices=[0], time_value1=frame_ini,
                                   time_value2=frame_end,
                                   thr_val=0.9,
                                   min_seed_dist=5, path_to_weights=path_to_weights, device='cuda')

        print('\n Segmentation step done')
        print("Post-processing of the segmentation : overwriting fluor. frames")

        fluorescence.remove_fluor_frames(path_to_full_movie,
                                         path_to_first_mask,
                                         filename,
                                         fluor_offset,
                                         fluor_step)
    print('\n')

    # ###################################
    # # step 2 : separate the traps from the movie
    # ###################################

    posttreatment.check_segmentation(path_to_full_movie, path_to_first_mask, './output')
    path_to_no_fluor_movie = './output/' + filename + '_no_fluor.tif'
    path_to_no_fluor_mask = './output/' + filename + "_no_fluor.h5"

    output_folder = './output'

    path_to_splitted_traps = template_matching.main(
        path_to_no_fluor_movie,
        path_to_no_fluor_mask,
        path_to_template,
        output_folder
    )

    print("\n")

    ###################################
    # step 3 : adapt the output format for midap
    ###################################

    if not args.no_format_midap:
        adapt_output_to_midap(path_to_splitted_traps, frame_ini, frame_end)

    print("\n")
    # ###################################
    # # step 4 : Tracking
    # ###################################

    if not args.no_tracking:
        list_traps = glob.glob(str(path_to_splitted_traps) + '/split_data/*')
        for i, trap_path in enumerate(list_traps):
            print(f"Tracking of trap {trap_path:s}")
            imgs = np.sort(glob.glob(trap_path + '/midap/cut_im/*.png'))
            segs = np.sort(glob.glob(trap_path + '/midap/seg_im/*.tif'))

            try:
                bst = BayesianCellTracking(imgs=imgs, segs=segs, model_weights=None)
                data_file, csv_file = bst.track_all_frames(trap_path + '/midap')
                print("Tracking saved in ", data_file, csv_file)
            except Exception as e:
                print('\n')
                print(f"Failed for trap {trap_path:s}")
                print(e)
                print('\n')
                pass

    print("\n")
    ###################################
    # step 5 : identify the mother cell
    ###################################

    if not args.no_mother:
        list_traps = glob.glob(str(path_to_splitted_traps) + '/split_data/*')
        for i, trap_path in enumerate(list_traps):
            print(f'Finding mother cell for {trap_path:s}')
            try:
                posttreatment.id_the_mother(trap_path, trap_center_size)
            except FileNotFoundError:
                print(f"File not found for {trap_path:s}. Is the trap empty for all frames?")
                pass

    ###################################
    # step 6 : Get the fluorescence values of each mother cell
    ###################################

    if not args.no_fluor:
        list_traps = glob.glob(str(path_to_splitted_traps) + '/split_data/*')
        for i, trap_path in enumerate(list_traps):
            print(f'Treating fluorescence data from {trap_path:s}')

            try:
                fluorescence.get_fluorescence_data(folder_init_movie,
                                                   trap_path,
                                                   Nmax, fluor_offset, fluor_step)
                posttreatment.generate_summary_plots(trap_path)
            except FileNotFoundError:
                print(f"File not found for {trap_path:s}")
                pass

    posttreatment.summary_csv(path_to_splitted_traps)

if __name__ == '__main__':
    main()
