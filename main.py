import template_matching

import time
import tqdm
from dotenv import load_dotenv
import sys

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
import pandas as pd
from skimage.registration import phase_cross_correlation
from scipy import ndimage
from tifffile import TiffWriter, imwrite

import matplotlib.pyplot as plt


# arguments from argparse
# '--image_path', type=str, help="Specify the path to a single image or to a folder of images", required=True)
# '--mask_path', type=str, help="Specify where to save predicted masks", required=True)
# '--image_type', type=str, help="Specify the imaging type, possible types are 'bf', 'pc', and 'fission'. Supersedes path_to_weights.")
# '--path_to_weights', default=None, type=str, help="Specify weights path.")
# '--fov', default=[0], nargs='+', type=int, help="Specify field of view index (can specify more than one with space between them).")
# '--range_of_frames', nargs=2, default=[0, 0], type=int, help="Specify start and end in range of frames. (e.g. 0 10)")
# '--threshold', default=None, type=float, help="Specify threshold value.")
# '--min_seed_dist', default=5, type=int, help="Specify minimum distance between seeds.")
# '--device', default='cpu', type=str, help="Specify device to run on (cpu or cuda).")


def LaunchPrediction(im, mic_type, pretrained_weights=None, device='cpu'):
    """It launches the neural neutwork on the current image and creates
    an hdf file with the prediction for the time T and corresponding FOV.
    """
    im = skimage.exposure.equalize_adapthist(im)
    im = im * 1.0;
    pred = nn.prediction(im, mic_type, pretrained_weights, device=device)
    return pred


def ThresholdPred(thvalue, pred):
    """Thresholds prediction with value"""
    if thvalue == None:
        thresholdedmask = nn.threshold(pred)
    else:
        thresholdedmask = nn.threshold(pred, thvalue)
    return thresholdedmask


def LaunchInstanceSegmentation(reader, image_type, fov_indices=[0], time_value1=0, time_value2=0, thr_val=None,
                               min_seed_dist=5, path_to_weights=None, device='cpu'):
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


def id_the_mother(path):
    # first, read the necessary input datas.
    # The pandas package seems a convenient option for the csv file

    tracking_table = pd.read_csv(path + '/midap/track_output_bayesian.csv')
    tracking_table['Centroid_x'] = np.nan
    tracking_table['Centroid_y'] = np.nan

    with h5py.File(path + '/midap/segmentations_bayesian.h5', "r") as f:
        a_group_key = list(f.keys())
        dset = f[a_group_key[0]]
        masks = list(dset)

    trap_center_size = [21, 41]
    image_size = np.shape(masks[0])
    trap_center_upper_left = [(image_size[0] - trap_center_size[0]) / 2,
                              (image_size[1] - trap_center_size[1]) / 2]

    # in one loop, line by line from pandas :
    # compute centroids of each cell in each frame
    # see if cell inside the trap
    # re-save the dataframe

    for index, row in tracking_table.iterrows():
        current_cell_id = int(row['labelID'])
        current_frame = int(row['frame'])

        frame_data = masks[current_frame]
        indices_cell = np.argwhere(frame_data == current_cell_id)

        centroids = [np.average(indices_cell[:, 0]), np.average(indices_cell[:, 1])]

        if (trap_center_upper_left[0] < centroids[0] < trap_center_upper_left[0] + trap_center_size[0] and
                trap_center_upper_left[1] < centroids[1] < trap_center_upper_left[1] + trap_center_size[1]):
            class_cell = 'trapped'
        else:
            class_cell = 'outsider'

        # put all of this nice data back in the df
        tracking_table.at[index, 'Centroid_x'] = centroids[1]
        tracking_table.at[index, 'Centroid_y'] = centroids[0]
        tracking_table.at[index, 'Type'] = class_cell

    # determine the cell that is inside the trap
    # for the highest number of frames
    unique_cells = tracking_table['trackID'].unique()

    max_frames = 0
    mother = -1
    index_list_save = []
    for cell in unique_cells:
        index_list = list(tracking_table.loc[(tracking_table['trackID'] == cell) &
                                             (tracking_table['Type'] == 'trapped')].index)

        if len(index_list) > max_frames:
            mother = cell
            max_frames = len(index_list)
            index_list_save = index_list

    for index in index_list_save:
        tracking_table.loc[[index], 'Type'] = "mother"

    tracking_table.to_csv(path + '/tracking_with_mother.csv')
    # this is the mother cell - you have won this game \o/


def correct_shift_in_mask(mask, list_shifts):
    with h5py.File(mask, "r+") as f_masks:
        for group in f_masks.keys():
            for dset in f_masks[group].keys():
                ds_data = f_masks[group][dset][:]

                frame_nb = int(dset.split('T')[-1])

                shifted_mask = ndimage.shift(ds_data, list_shifts[frame_nb])
                f_masks[group][dset][...] = shifted_mask


def correct_shift(path_to_image, mask):
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

    return list_of_shifted_images


def get_fluorescence_data(trap_path, Nmax, fluor_offset, fluor_step):
    print(trap_path)
    track_df = pd.read_csv(trap_path + '/tracking_with_mother.csv')
    list_fluor_frames = np.linspace(fluor_offset,
                                    Nmax - 1,
                                    int((Nmax - fluor_offset) / fluor_step) + 1)

    h5data_path = glob.glob('./input/example_movie/*.h5')
    data_fluor = h5py.File(h5data_path[0], 'r')

    tifdata_path = glob.glob('./input/example_movie/*.tif')
    im = io.imread(tifdata_path[0])

    group = list(data_fluor.keys())[0]

    # get cell id in the frame before the fluor frame
    # get the list of fluor px of this cell
    # write in csv file the avg and std-dev and the list of px
    # move on with life

    track_df['fluor_avg'] = -1.
    track_df['fluor_max'] = -1.
    track_df['fluor_std'] = -1.
    track_df['fluorescence_values'] = 'none'
    for i_frame in list_fluor_frames:
        i_mask = int(i_frame - 1.)
        i_fluor = int(i_frame)

        cell_list = list(track_df.loc[track_df['frame'] == i_mask]['labelID'])

        dset = "T" + str(i_mask)
        arr = data_fluor[group][dset][:, :]

        for cellid in cell_list:

            indices = np.where(arr == cellid)

            fluorescence_per_cell = []
            for i in range(0, len(indices[0])):
                ind1 = indices[0][i]
                ind2 = indices[1][i]
                fluorescence_per_cell.append(im[i_fluor][ind1, ind2])

            track_df.loc[(track_df['frame'] == int(i_frame)) &
                         (track_df['labelID'] == cellid),
                         'fluor_avg'] = np.average(fluorescence_per_cell)

            track_df.loc[(track_df['frame'] == int(i_frame)) &
                         (track_df['labelID'] == cellid),
                         'fluor_max'] = np.max(fluorescence_per_cell)

            track_df.loc[(track_df['frame'] == int(i_frame)) &
                         (track_df['labelID'] == cellid),
                         'fluor_std'] = np.std(fluorescence_per_cell)

            track_df.loc[(track_df['frame'] == int(i_frame)) &
                         (track_df['labelID'] == cellid),
                         'fluorescence_values'] = str(fluorescence_per_cell)

    track_df.to_csv(trap_path + '/tracking_with_fluor.csv')


def generate_summary_plots(trap_path):
    trap_nb = trap_path.split('/')[-1]

    full_df = pd.read_csv(trap_path + '/tracking_with_fluor.csv')
    df_inter = full_df.loc[(full_df['Type'] == 'mother') & (full_df['fluor_avg'] != -1.)]
    df_fluor_mother = df_inter[['frame', 'fluor_avg', 'fluor_std']].copy()
    df_fluor_val_mother = df_inter[['frame', 'fluorescence_values']].copy()

    fluor_stat_arr = df_fluor_mother.to_numpy()
    fluor_val_arr = df_fluor_val_mother.to_numpy()
    # the second value is a string, but we only need
    # the number of elements separated by the comma
    # = number of fluor px to get an indication on the cell size :

    size_cell = lambda x: [len(elmt[1].split(',')) for elmt in x]

    to_plot_size_cell = np.vstack((fluor_val_arr[:, 0],
                                   size_cell(fluor_val_arr)[:]))

    plt.rcParams["figure.figsize"] = (12, 10)
    fig, axs = plt.subplots(2)

    l1, = axs[0].plot(fluor_stat_arr[:, 0], fluor_stat_arr[:, 1])
    l2, = axs[0].plot(fluor_stat_arr[:, 0], fluor_stat_arr[:, 2])
    axs[0].set_title('Fluorescence values, trap nb. ' + trap_nb)
    axs[0].set_xlabel('Frame number')
    axs[0].set_ylabel('Average value or std, units from tif file')
    fig.legend((l1, l2), ('Average', 'Standard deviation'))

    axs[1].plot(to_plot_size_cell[0, :], to_plot_size_cell[1, :], color='tab:green')
    axs[1].set_title('Size of the cell')
    axs[1].set_xlabel('Frame number')
    axs[1].set_ylabel('Size of cell (number of px)')
    plt.savefig(trap_path + '/summary_trap_' + trap_nb + '.png')
    plt.close()

    return 0


def remove_fluor_frames(path_to_full_movie,
                        path_to_first_mask,
                        filename,
                        fluor_offset,
                        fluor_step):
    im = io.imread(path_to_full_movie)
    data = h5py.File(path_to_first_mask, "r+")
    data_no_fluor = h5py.File('./output/' + filename + "_no_fluor.h5", "w")

    for group in data.keys():
        g1 = data_no_fluor.create_group(group)
        for i in range(0, len(data[group].keys())):
            dset = 'T' + str(i)

            arr = data[group][dset][:, :]

            if i == 0:
                arr_old = arr

            frame_num = int(dset.split("T")[1])

            if (frame_num - fluor_offset) % fluor_step == 0 and frame_num >= fluor_offset:
                # we are going to duplicate the previous picture and replace the fluor. picture
                im[frame_num] = im[frame_num - 1]
                g1.create_dataset(dset, data=arr_old)
                data_no_fluor[group][dset][:, :] = arr_old
            else:
                g1.create_dataset(dset, data=arr)
                pass

            arr_old = arr

    im_nofluor = im
    imwrite('./output/' + filename + '_no_fluor.tif', im_nofluor)
    data_no_fluor.close()


def adapt_output_to_midap(root_path, frame_ini, frame_end):
    list_traps = glob.glob(root_path + '/*')
    list_traps.sort()

    for i, trap_path in enumerate(list_traps):
        print(f"Converting output format of trap {trap_path:s}")

        maskfile = glob.glob(list_traps[i] + '/*h5')[0]
        tiffile = glob.glob(list_traps[i] + '/*tif')[0]

        list_shifted_im = correct_shift(tiffile, maskfile)
        list_shifted_im[0].save(list_traps[i] + '/shift_corrected.tif',
                                save_all=True,
                                append_images=list_shifted_im[1:])

        tiffile = list_traps[i] + '/shift_corrected.tif'

        Path(trap_path + '/midap/seg_im').mkdir(parents=True, exist_ok=True)
        Path(trap_path + '/midap/cut_im').mkdir(parents=True, exist_ok=True)

        mask = h5py.File(maskfile, "r")
        group = "FOV0"
        list_dset_num = range(frame_ini, frame_end)

        for n in list_dset_num:
            dset = 'T' + str(n)
            arr = mask[group][dset][:]
            arr = arr.astype(np.uint16)

            tiffwrite(trap_path + '/midap/seg_im/im_' + f"{n:03d}" + '.tif', arr, 'XY')

        im = io.imread(tiffile)
        for n in range(frame_ini, frame_end):
            im_for_png = Image.fromarray(im[n])
            im_for_png.save(trap_path + "/midap/cut_im/im_" + f"{n:03d}" + ".png")


def arguments():
    """Parsing the input arguments"""

    parser = argparse.ArgumentParser(
        description="Run the segmentation and tracking pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # parser.add_argument(
    #     "-i",
    #     "--input_image",
    #     help="Input image including the full path for extraction",
    #     type=str,
    #     required=True
    # )
    # parser.add_argument(
    #     "-t",
    #     "--template_image",
    #     help="Template image including the full path for extraction",
    #     type=str,
    #
    # )

    parser.add_argument(
        "-no_s",
        "--no_segmentation",
        help="Exclude the segmentation step from the pipeline",
        default=False,
        action='store_true'
    )

    parser.add_argument(
        "-no_spl",
        "--no_splitting_traps",
        help="Exclude the step that splits the main movie by trap",
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

    return parser.parse_args()


def main():
    fluor_step = 6  # this should be an argument
    fluor_offset = 9  # idem

    load_dotenv()

    path_to_full_movie = os.getenv('FULL_MOVIE_YEAST')
    path_to_weights = os.getenv('WEIGHTS_YEAST')
    path_to_template = os.getenv('TEMPLATE_YEAST_TRAP')

    path_to_first_mask = path_to_full_movie.split(".tif")[0]
    path_to_first_mask = path_to_first_mask + '.h5'
    filename = path_to_full_movie.split('/')[-1]
    filename = filename.split('.')[0]

    args = arguments()

    im = io.imread(path_to_full_movie)
    Nmax = im.shape[0]
    frame_ini = 0
    frame_end = Nmax - 1
    im.close()

    ###################################
    # step 1 : do the segmentation of the full movie [code from launch_NN_commandline]
    # Including the fluor frames. Ignore fluor segmentation and separate the fluor
    # from non-fluor frames in a post-processing step
    ###################################

    if not args.no_segmentation:
        reader = nd.Reader("", path_to_first_mask, path_to_full_movie)
        LaunchInstanceSegmentation(reader, image_type=None, fov_indices=[0], time_value1=frame_ini,
                                   time_value2=frame_end,
                                   thr_val=0.9,
                                   min_seed_dist=5, path_to_weights=path_to_weights, device='cuda')

        print('Segmentation step done')
        print("Post-processing of the segmentation : overwriting fluor. frames")

        remove_fluor_frames(path_to_full_movie,
                            path_to_first_mask,
                            filename,
                            fluor_offset,
                            fluor_step)

    print('\n')

    # ###################################
    # # step 2 : separate the traps from the movie
    # ###################################

    path_to_no_fluor_movie = './output/' + filename + '_no_fluor.tif'
    path_to_no_fluor_mask = './output/' + filename + "_no_fluor.h5"

    if not args.no_splitting_traps:
        os.system(
            'python template_matching.py -i ' + path_to_no_fluor_movie +
            ' -t ' + path_to_template +
            ' -o ./output' +
            ' -im ' + path_to_no_fluor_mask)

    print("\n")

    ###################################
    # step 3 : adapt the output format for midap
    ###################################


    if not args.no_format_midap:
        adapt_output_to_midap(path_to_splitted_traps)

    print("\n")
    # ###################################
    # # step 4 : Tracking
    # ###################################

    if not args.no_tracking:
        list_traps = glob.glob(path_to_splitted_traps + '/*')
        for i, trap_path in enumerate(list_traps):
            print(f"Tracking of trap {trap_path:s}")
            imgs = np.sort(glob.glob(trap_path + '/midap/cut_im/*.png'))
            segs = np.sort(glob.glob(trap_path + '/midap/seg_im/*.tif'))

            bst = BayesianCellTracking(imgs=imgs, segs=segs, model_weights=None)
            data_file, csv_file = bst.track_all_frames(trap_path + '/midap')

            # st = STrack(imgs=imgs, segs=segs, model_weights=None)
            # data_file, csv_file = st.track_all_frames(trap_path+'/midap', max_dist=10.0, max_angle=180.0)

            print("Tracking saved in ", data_file, csv_file)

    print("\n")
    ###################################
    # step 5 : identify the mother cell
    ###################################

    if not args.no_mother:
        list_traps = glob.glob(path_to_splitted_traps + '/*')
        for i, trap_path in enumerate(list_traps):
            id_the_mother(trap_path)

    ###################################
    # step 6 : Get the fluorescence values of each mother cell
    ###################################

    list_traps = glob.glob(path_to_splitted_traps + '/*')
    for i, trap_path in enumerate(list_traps):
        get_fluorescence_data(trap_path, Nmax, fluor_offset, fluor_step)
        generate_summary_plots(trap_path)


if __name__ == '__main__':
    main()
