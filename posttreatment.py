import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import glob
from PIL import Image, ImageSequence
import math
import cv2


def id_the_mother(path: str, trap_center_size: list) -> None:
    '''
    Uses the csv file resulting from the tracking step to identify the mother cell;
    The criteria being that the mother cell is the cell that stays in the center
    of the trap for the longest time in the entire movie. As it is a "max" criteria
    on the number of frames, there is only one mother cell per trap for the entire movie

    path : path to the folder containing splitted traps
    trap_center_size : size, in px, of the center of the trap
    '''

    # first, read the necessary input datas.
    # The pandas package seems a convenient option for the csv file

    tracking_table = pd.read_csv(path + '/midap/track_output_bayesian.csv')
    tracking_table['Centroid_x'] = np.nan
    tracking_table['Centroid_y'] = np.nan

    with h5py.File(path + '/midap/segmentations_bayesian.h5', "r") as f:
        a_group_key = list(f.keys())
        dset = f[a_group_key[0]]
        masks = list(dset)

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
    index_list_save = []
    for cell in unique_cells:
        index_list = list(tracking_table.loc[(tracking_table['trackID'] == cell) &
                                             (tracking_table['Type'] == 'trapped')].index)

        if len(index_list) > max_frames:
            max_frames = len(index_list)
            index_list_save = index_list

    for index in index_list_save:
        tracking_table.loc[[index], 'Type'] = "mother"

    tracking_table.to_csv(path + '/tracking_with_mother.csv')


def generate_summary_plots(trap_path: str) -> None:
    '''
    Will generate two plats per trap : the evolution of the size of the
    mother cell with frame, and the evolution of the average fluorescence
    value with frame.

    trap_path : path to the directory containing the movie in splitted traps
    '''
    trap_nb = trap_path.split('/')[-1]

    full_df = pd.read_csv(trap_path + '/tracking_with_fluor.csv')
    df_inter = full_df.loc[(full_df['Type'] == 'mother') & (full_df['fluor_avg'] != -1.)]
    df_fluor_mother = df_inter[['frame', 'fluor_avg', 'fluor_std']].copy()
    df_fluor_val_mother = df_inter[['frame', 'fluorescence_values']].copy()

    fluor_stat_arr = df_fluor_mother.to_numpy()
    fluor_val_arr = df_fluor_val_mother.to_numpy()

    list_of_fluor_arrays = []
    for numframe, values in fluor_val_arr:
        reconstructed_list = []
        for val in values[1:-1].split(","):
            reconstructed_list.append(float(val))
        list_of_fluor_arrays.append(reconstructed_list)

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


def full_movie_for_quality_check(path_to_splitted_traps):
    # list_traps = glob.glob(str(path_to_splitted_traps) + '/split_data/*')
    #
    # N = int( math.sqrt(len(list_traps)) )
    #
    # for trap in list_traps :
    #     data = Image.open(trap+'/shift_corrected.tif')
    #     frames = np.array(data)
    #
    #     with h5py.File(trap + '/midap/segmentations_bayesian.h5', "r") as f:
    #         a_group_key = list(f.keys())
    #         dset = f[a_group_key[0]]
    #         masks = list(dset)
    #
    #         for i, mask in enumerate(masks):
    #
    #
    #     #combine mask and tif
    #
    #     #append the image to the main image
    #

    return 0


def summary_csv(path_to_splitted_traps):
    list_traps = glob.glob(str(path_to_splitted_traps) + '/split_data/*')

    all_df = []
    cpt_id = 0
    old_id = 1
    for trap in list_traps:
        trap_nb = int(trap.split('/')[-1])
        one_df = pd.read_csv(trap + '/tracking_with_fluor.csv')
        one_df['trap_nb'] = trap_nb

        one_df['Global_id'] = 0

        for index, row in one_df.iterrows():

            if row['trackID'] != old_id:
                cpt_id += 1
                old_id = row['trackID']
                one_df.loc[one_df['trackID'] == old_id, 'Global_id'] = cpt_id

        all_df.append(one_df)

    df_res = pd.concat(all_df, ignore_index=True)
    df_res.to_csv(path_to_splitted_traps + '/all_traps_summary.csv')
