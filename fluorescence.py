import h5py
from skimage import io
from tifffile import imwrite
import pandas as pd
import numpy as np
import glob


def detect_fluor_frames(movie, min_ratio: float = 2.0) -> list:
    """Frame indices holding a fluorescence image, worked out from intensity alone.

    A fluorescence frame looks nothing like a bright-field one: on the example
    movies the per-frame median is ~640 against ~28300, while the bright-field
    frames agree with each other to better than 1%. So sorting the per-frame
    medians leaves exactly one enormous gap, and the split needs no tuned
    threshold - take the largest ratio between neighbouring sorted values, and if
    it exceeds min_ratio the two sides are different kinds of image.

    The smaller group is taken to be the fluorescence one, which also covers
    fluorescence that is *brighter* than the bright-field rather than darker.

    movie : 3D array (time, y, x)
    min_ratio : how many times apart the two groups must be to count as
                different kinds of image. Returns [] below this, i.e. the movie
                is treated as bright-field throughout. 2 is already far outside
                anything ordinary illumination drift produces, the bright-field
                frames of the example movies agreeing to within 1%, and leaves
                room for a fluorescence channel that is only a little brighter
                than the bright-field rather than much darker.
    """
    medians = np.array([np.median(frame) for frame in movie], dtype=float)
    if len(medians) < 3:
        return []

    order = np.sort(medians)
    # guard against a zero median making the ratio meaningless
    safe = np.maximum(order, 1e-9)
    ratios = safe[1:] / safe[:-1]

    split = int(np.argmax(ratios))
    if ratios[split] < min_ratio:
        return []

    # anything at or below the gap is one kind of image, above it the other
    threshold = np.sqrt(safe[split] * safe[split + 1])
    below = np.flatnonzero(medians <= threshold)
    above = np.flatnonzero(medians > threshold)

    fluor = below if len(below) <= len(above) else above
    return sorted(int(f) for f in fluor)


def fluor_offset_and_step(frames: list, Nmax: int) -> tuple:
    """(offset, step) describing a list of fluorescence frames.

    Returns (None, None) when there are none. `step` is the most common spacing;
    it is None when there is only one fluorescence frame, in which case any step
    past the end of the movie describes it equally well.
    """
    if not frames:
        return None, None

    offset = int(frames[0])
    if len(frames) == 1:
        return offset, None

    spacings, counts = np.unique(np.diff(frames), return_counts=True)
    step = int(spacings[np.argmax(counts)])

    return offset, step


def list_fluor_frames(Nmax: int, fluor_offset: int, fluor_step: int) -> list:
    """
    Frame indices of the fluorescence images in a movie of Nmax frames.

    Nmax : total number of frames in the movie
    fluor_offset : number of frames before the first fluorescence frame
    fluor_step : number of frames between two fluorescence frames
    """
    if fluor_step <= 0:
        raise ValueError(f"fluor_step must be positive, got {fluor_step}")
    return list(range(fluor_offset, Nmax, fluor_step))


def overwrite_fluor_frames(path_to_full_movie: str,
                           path_to_no_fluor_movie: str,
                           frames) -> list:
    """
    Write a copy of the movie in which every fluorescence frame has been
    replaced by the bright-field frame immediately before it, and return the
    list of frames that were replaced.

    This runs before the segmentation, so the network only ever sees
    bright-field data.

    path_to_full_movie : path to the initial input movie
    path_to_no_fluor_movie : where to write the bright-field-only movie
    frames : indices of the fluorescence frames
    """
    im = io.imread(path_to_full_movie)
    frames = sorted(frames)

    for frame_num in frames:
        if frame_num == 0:
            # nothing before it to copy; fall back to the first bright-field frame
            following = [f for f in range(1, len(im)) if f not in frames]
            if following:
                im[0] = im[following[0]]
            continue
        im[frame_num] = im[frame_num - 1]

    imwrite(path_to_no_fluor_movie, im)

    return frames


def get_fluorescence_data(path_to_init_movie: str,
                          path_to_first_mask: str,
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

    # range() stops before Nmax, so the last entry can never point past the end
    # of the movie the way the old hand-rolled loop could
    fluor_frames = list_fluor_frames(Nmax, fluor_offset, fluor_step)

    h5data_path = path_to_first_mask #glob.glob(initial_movie_path + '/*.h5')
    data_fluor = h5py.File(h5data_path, 'r')

    tifdata_path = path_to_init_movie #glob.glob(initial_movie_path + '/*.tif')
    im = io.imread(tifdata_path)

    group = list(data_fluor.keys())[0]

    # get cell id in the frame before the fluor frame
    # get the list of fluor px of this cell
    # write in csv file the avg and std-dev and the list of px
    # move on with life

    track_df['fluor_avg'] = -1.
    track_df['fluor_max'] = -1.
    track_df['fluor_std'] = -1.
    track_df['fluorescence_values'] = 'none'

    for i_frame in fluor_frames:
        i_mask = int(i_frame - 1.)
        i_fluor = int(i_frame)

        if i_mask < 0:
            # first frame of the movie is a fluorescence frame, so there is no
            # preceding bright-field mask to measure through
            continue

        cell_list = list(track_df.loc[track_df['frame'] == i_mask]['labelID'])

        dset = "T" + str(i_mask)
        arr = data_fluor[group][dset][:, :]
        fluor_image = im[i_fluor]

        for cellid in cell_list:

            # gather the cell's pixels in one go rather than one at a time
            fluorescence_per_cell = fluor_image[arr == cellid].tolist()

            if not fluorescence_per_cell:
                continue

            rows = ((track_df['frame'] == int(i_frame)) &
                    (track_df['labelID'] == cellid))

            track_df.loc[rows, 'fluor_avg'] = np.average(fluorescence_per_cell)
            track_df.loc[rows, 'fluor_max'] = np.max(fluorescence_per_cell)
            track_df.loc[rows, 'fluor_std'] = np.std(fluorescence_per_cell)
            track_df.loc[rows, 'fluorescence_values'] = str(fluorescence_per_cell)

    data_fluor.close()
    track_df.to_csv(trap_path + '/tracking_with_fluor.csv')