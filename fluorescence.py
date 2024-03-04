import h5py
from skimage import io
from tifffile import imwrite
import pandas as pd
import numpy as np
import glob


def remove_fluor_frames(path_to_full_movie: str,
                        path_to_first_mask: str,
                        filename: str,
                        fluor_offset: int,
                        fluor_step: int) -> None:
    """
    This routine will create a new movie and a new maskfile, where all the fluorescence
    frames have been *overwritten* by the frame that precedes them.

    path_to_full_movie : path to the initial input movie
    path_to_first_mask : path to the first mask created by the segmentation step
    filename : name of the initial file (suffix removed)
    fluor_offset : number of frames before the first frame
    fluor_step : number of frames in between fluorescence frames
    """

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


def get_fluorescence_data(initial_movie_path: str,
                          trap_path: str,
                          Nmax: int,
                          fluor_offset: int,
                          fluor_step: int) -> None:
    """
    This routine will extract the fluorescence data : it will go and grab the
    fluorescence frames and mask from the initial movie, and then it will
    apply the mask of the previous frame to the fluorescence frame. All pixels
    inside the mask are extracted and saved, with additional values such as avg, std, etc.

    trap_path : path to the trap of interest
    Nmax : total number of frames in movie
    fluor_offset : number of frames before the first fluorescence frames
    fluor_step : number fo frames in between each fluorescence frame
    """
    track_df = pd.read_csv(trap_path + '/tracking_with_mother.csv')

    list_fluor_frames = [fluor_offset]
    cpt = fluor_offset

    while cpt <= Nmax - fluor_step:
        list_fluor_frames.append(list_fluor_frames[-1] + fluor_step)
        cpt += fluor_step

    h5data_path = glob.glob(initial_movie_path + '*.h5')
    data_fluor = h5py.File(h5data_path[0], 'r')

    tifdata_path = glob.glob(initial_movie_path + '*.tif')
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