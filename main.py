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
import shutil
import h5py
from tiffwrite import tiffwrite
from pathlib import Path
from datetime import datetime
from skimage import io
from PIL import Image, ImageSequence

import btrack
from btrack.constants import BayesianUpdates
from midap.tracking import bayesian_tracking
from midap.tracking.bayesian_tracking import BayesianCellTracking

import numpy as np
import pandas as pd
from skimage.registration import phase_cross_correlation
from skimage.measure import regionprops
from scipy import ndimage
import drift
import posttreatment
import fluorescence
import trap_detection

# Objects smaller than this (in px) are segmentation debris rather than cells.
# They are removed before tracking, where they otherwise steal links from real
# cells and spawn large numbers of 1-2 frame long spurious tracks.
MIN_CELL_AREA = 50

# Region props handed to btrack and used by its VISUAL update to score links.
TRACK_FEATURES = ('area', 'major_axis_length', 'minor_axis_length',
                  'orientation', 'solidity', 'eccentricity')


def pick_device(requested='cuda'):
    """Return the fastest torch device actually available.

    The pipeline used to ask yeaz for 'cuda' and silently fall back to the CPU
    whenever CUDA was missing, which on an Apple-silicon Mac meant never using
    the GPU at all. Metal (mps) is ~14x faster than the CPU here for the same
    numerical result.
    """
    if requested == 'cpu':
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def normalise_frame(im):
    """Contrast-equalise a frame the way the YeaZ weights expect.

    NOTE: this movie segments noticeably differently between frames 10-20 and
    the rest (~30% more objects, median area 226 vs 281), tracking a change in
    the frames' intensity range, so the preprocessing here looks like a real
    lead for further segmentation gains. Rescaling each frame to its own
    1st/99th percentile before equalising does even out the object counts - but
    it does so by clipping the tails that carry the cell edges, which shrinks
    real cells and makes some disappear entirely. Anything tried here needs to
    be checked on cell *areas*, not just object counts.
    """
    return skimage.exposure.equalize_adapthist(im) * 1.0


class Segmenter:
    """Runs the YeaZ U-Net, keeping the model resident on the chosen device.

    yeaz.unet.neural_network.prediction rebuilds the model and re-reads the
    124 MB weights file on every single call, and can only ever use the CPU or
    CUDA. Holding the model here instead lets us use Metal and skip the
    per-frame reload.
    """

    def __init__(self, path_to_weights, device='cuda'):
        if not path_to_weights or not os.path.exists(str(path_to_weights)):
            raise FileNotFoundError(
                f"Could not find the YeaZ weights at {path_to_weights!r}. "
                "Download them as described in the README and point the "
                "WEIGHTS_YEAST entry of your .env file at the file.")

        from yeaz.unet.model_pytorch import UNet

        self.device = pick_device(device)
        self.model = UNet()
        self.model.load_state_dict(torch.load(str(path_to_weights),
                                              map_location='cpu'))
        self.model = self.model.to(self.device).eval()

    def predict(self, im):
        """Probability map for one frame, same convention as nn.prediction."""
        nrow, ncol = im.shape
        padded = np.pad(im, ((0, 16 - nrow % 16), (0, 16 - ncol % 16)))
        tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            out = self.model(tensor.to(self.device)).cpu().numpy()
        return out[0, 0, :nrow, :ncol]

    def segment_frame(self, im, thr_val=None, min_seed_dist=5):
        """Normalise, predict, threshold and split one frame into labelled cells."""
        pred = self.predict(normalise_frame(im))
        thresh = nn.threshold(pred, thr_val) if thr_val is not None else nn.threshold(pred)
        return segment(thresh, pred, min_seed_dist)


def LaunchInstanceSegmentation(reader, image_type,
                               fov_indices=None, time_value1=0, time_value2=0,
                               thr_val=None,
                               min_seed_dist=5,
                               path_to_weights=None, device='cpu',
                               duplicate_frames=(), use_correspondence=False):
    """Segment every frame of the movie and store the masks through `reader`.

    duplicate_frames : frames known to be byte-identical copies of the frame
                       before them (the overwritten fluorescence frames). Their
                       mask is copied rather than recomputed, which saves one
                       network pass per fluorescence frame.
    use_correspondence : run YeaZ's frame-to-frame label matching. Off by
                       default: it renumbers cells so that a label means the same
                       cell from one frame to the next, but nothing downstream
                       uses that. btrack re-tracks from the raw masks, and the
                       fluorescence, mother and summary steps only ever match a
                       label within a single frame. It costs more time than the
                       network itself - a pure-Python Hungarian match plus a
                       full-image pass per cell, on ~150 cells a frame.
    """
    if fov_indices is None:
        fov_indices = [0]

    # cannot have both path_to_weights and image_type supplied
    if (image_type is not None) and (path_to_weights is not None):
        raise ValueError("image_type and path_to_weights cannot be both supplied.")

    # check if correct imaging value
    if (image_type not in ['bf', 'pc']) and (path_to_weights is None):
        raise ValueError(
            f"Wrong imaging type value ({image_type!r}); imaging type must be "
            "either 'bf' or 'pc', or a path to custom weights must be given.")

    # check range_of_frames constraint
    if time_value1 > time_value2:
        raise ValueError('Invalid time constraints: '
                         f'{time_value1} > {time_value2}')

    segmenter = Segmenter(path_to_weights, device=device)
    duplicate_frames = set(duplicate_frames)

    print('Running the neural network on {} ...'.format(segmenter.device.type))

    for fov_ind in tqdm.tqdm(fov_indices, desc='FOV', position=0):

        previous_seg = None
        # iterates over the time indices in the range
        for t in tqdm.tqdm(range(time_value1, time_value2 + 1), desc='Time', position=1, leave=False):

            if t in duplicate_frames and previous_seg is not None:
                # identical image, so identical mask - no need to ask the network
                seg = previous_seg
            else:
                im = reader.LoadOneImage(t, fov_ind)
                seg = segmenter.segment_frame(im, thr_val=thr_val,
                                              min_seed_dist=min_seed_dist)

            reader.SaveMask(t, fov_ind, seg)

            if use_correspondence:
                # renumber so a label means the same cell as in the frame before
                temp_mask = reader.CellCorrespondence(t, fov_ind)
                reader.SaveMask(t, fov_ind, temp_mask)
            previous_seg = seg


def correct_shift_in_mask(mask: str, list_shifts: dict) -> None:
    '''
    Correct the drift/shift of the movie on maskfiles.
    Will *replace* the original mask with the corrected mask

    mask : path to mask file to correct
    list_shifts : dictionary containing the shifts per frame
    '''
    with h5py.File(mask, "r+") as f_masks:
        if f_masks.attrs.get('shift_corrected'):
            # correct_shift_in_mask rewrites the file in place, so applying it a
            # second time (re-running with -no_s against an existing output
            # folder) would shift the masks twice and pull them off the images
            print(f"  {mask} is already shift corrected, leaving it alone")
            return
        for group in f_masks.keys():
            for dset in f_masks[group].keys():
                ds_data = f_masks[group][dset][:]

                frame_nb = int(dset.split('T')[-1])

                # order=0: these are integer cell labels, so they must be moved
                # by nearest neighbour. Interpolating between them invents
                # label values that belong to no cell.
                shifted_mask = ndimage.shift(ds_data, np.round(list_shifts[frame_nb]),
                                             order=0, mode='constant', cval=0)
                f_masks[group][dset][...] = shifted_mask
        f_masks.attrs['shift_corrected'] = True


# NOTE: for_registration now LIVES IN drift.py. This is only an alias, kept so
# that existing callers here keep working.
#
# Editing it here will do nothing. Redefining it below this line would silently
# shadow the real one and be ignored by everything in drift.py - which is where
# the drift is actually estimated. Change drift.for_registration instead.
#
# It moved because drift estimation is useful on its own and needs only numpy,
# scipy and scikit-image, where importing this module costs ~6 s and pulls in
# torch, yeaz, midap and btrack.
for_registration = drift.for_registration


def correct_shift(path_to_image: str, mask: str = None, shifts=None) -> (list, dict):
    '''
    Will correct the shift/drift of the movie (stack tif file) with respect to the
    first frame. The corrected images are the output of the function.

    path_to_image : path to the image that needs shift correction
    mask : path to mask corresponding to the image. If given, it is corrected in
           place too. Optional, so that the images can be corrected on their own.
    shifts : pre-computed per-frame shifts (see estimate_global_drift). If None,
             the shifts are estimated from this crop alone - which is worse, as
             a crop full of cells has little fixed structure to lock on to.
    '''
    frames = ImageSequence.all_frames(Image.open(path_to_image))

    if shifts is None:
        shifts = drift.estimate_drift(frames)

    list_of_shifted_images = []
    list_shifts = {}
    for i, frame in enumerate(frames):
        shift = np.round(shifts[i])
        shifted = drift.apply_drift_to_frame(np.asarray(frame, dtype=np.uint16), shift)
        list_of_shifted_images.append(Image.fromarray(shifted))
        list_shifts[i] = shift

    if mask is not None:
        correct_shift_in_mask(mask, list_shifts)

    return list_of_shifted_images, list_shifts


def estimate_global_drift(path_to_image: str) -> np.ndarray:
    '''
    Estimate the drift of the whole field of view, frame by frame, relative to
    the first frame.

    Estimating the drift once on the full frame is far more reliable than
    estimating it separately inside each small trap crop: a trap that fills up
    with cells has little fixed structure left for the correlation to lock on
    to, which is what makes drift correction fail for individual traps.

    The measurement itself lives in `drift`, which needs nothing from this
    pipeline and so can be used on its own.

    path_to_image : path to the full (no-fluorescence) movie
    '''
    return drift.estimate_drift(ImageSequence.all_frames(Image.open(path_to_image)))


class EmptyTrap(Exception):
    """Raised for a trap that holds no cells, so there is nothing to track."""


class TrackingWithFeatures(BayesianCellTracking):
    """btrack tracking with the VISUAL update actually switched on.

    midap configures btrack with tracking_updates = ["VISUAL", "MOTION"] but
    never measures any object features, and btrack silently sets n_features = 0
    and skips the visual term - so links are scored on motion alone. Measuring a
    few shape properties and registering them as features makes the visual half
    of that configuration do something, which helps most where the motion model
    is weakest: cells packed together in a trap, barely moving, where position
    alone does not say which cell is which.

    This subclass lives here rather than in midap because the installed midap is
    a separate checkout shared with other projects.
    """

    features = TRACK_FEATURES

    def __init__(self, *args, duplicate_frames=(), **kwargs):
        """
        :param duplicate_frames: frames that are byte-identical copies of the
            frame before them (the overwritten fluorescence frames). These are
            withheld from the tracker and restored afterwards - see
            restore_duplicate_frames for why.
        """
        super().__init__(*args, **kwargs)

        self.duplicate_frames = sorted(set(duplicate_frames))
        self.num_time_steps_full = len(self.seg_imgs)
        self.tracked_frames = [t for t in range(self.num_time_steps_full)
                               if t not in set(self.duplicate_frames)]

        # keep the full stacks for the stored output, track on the rest
        self.seg_imgs_full, self.raw_imgs_full = self.seg_imgs, self.raw_imgs
        self.seg_imgs = self.seg_imgs[self.tracked_frames]
        self.raw_imgs = self.raw_imgs[self.tracked_frames]

    def track_all_frames(self, output_folder):
        if not self.seg_imgs_full.any():
            # An empty trap is a normal thing to find, not a failure. btrack
            # returns no tracks, midap then hands its numba label_transform an
            # empty 1-D array where it expects an (N, 3) one, and the run ends in
            # a NumbaTypeError that reads like a bug in the tracker.
            raise EmptyTrap('no cells were segmented anywhere in this trap')

        tracks = self.run_model()
        if not tracks:
            raise EmptyTrap('the tracker found no tracks in this trap')
        df, label_stack = self.generate_midap_output(tracks=tracks)
        df, label_stack = self.restore_duplicate_frames(df, label_stack)

        # everything downstream indexes by real frame number, so hand the full
        # stacks to the writer
        self.seg_imgs, self.raw_imgs = self.seg_imgs_full, self.raw_imgs_full

        return self.store_lineages(output_folder=output_folder, df=df,
                                   label_stack=label_stack)

    def restore_duplicate_frames(self, df, label_stack):
        """Put the withheld fluorescence frames back into the tracking output.

        A fluorescence frame is acquired alongside the bright-field frame before
        it, not at a time point of its own, so the two bright-field frames either
        side of it are one step apart and the tracker should simply not see the
        fluorescence frame at all. (Renumbering the objects to the original frame
        indices, which makes that gap look like two steps, is wrong and costs a
        lot: 372 -> 495 tracks over six traps.)

        A fluorescence frame carries no new information - its mask is a copy of
        the bright-field frame before it - but the tracker cannot know that. It
        sees every cell hold perfectly still for one step and then move two
        frames' worth on the next, which contradicts the constant-velocity motion
        model, and the global optimiser resolves the contradiction by handing a
        cell's identity to a neighbour. Measured on the 364 frame movie, 24.7% of
        mid-movie track starts landed on the frame straight after a fluorescence
        frame against 16.3% expected by chance.

        So the tracker only ever sees real bright-field frames, and each skipped
        frame then simply inherits the assignment of the frame it was copied
        from - which is exactly right, the masks being identical.
        """
        frame_of = {i: t for i, t in enumerate(self.tracked_frames)}

        df = df.copy()
        df['frame'] = df['frame'].astype(int).map(frame_of)

        duplicated = []
        for frame_nb in self.duplicate_frames:
            source = df[df['frame'] == frame_nb - 1]
            if len(source):
                copy = source.copy()
                copy['frame'] = frame_nb
                duplicated.append(copy)
        if duplicated:
            df = pd.concat([df] + duplicated, ignore_index=True)

        df = df.sort_values(['trackID', 'frame']).reset_index(drop=True)

        # the track's extent and the frame a split happens on have both moved
        spans = df.groupby('trackID')['frame']
        df['first_frame'] = df['trackID'].map(spans.min())
        df['last_frame'] = df['trackID'].map(spans.max())
        df['split'] = 0
        has_daughter = df['trackID_d1'].notna() | df['trackID_d2'].notna()
        df.loc[has_daughter & (df['frame'] == df['last_frame']), 'split'] = 1

        full_stack = np.zeros((self.num_time_steps_full,) + label_stack.shape[1:],
                              dtype=label_stack.dtype)
        for i, frame_nb in enumerate(self.tracked_frames):
            full_stack[frame_nb] = label_stack[i]
        for frame_nb in self.duplicate_frames:
            if frame_nb > 0:
                full_stack[frame_nb] = full_stack[frame_nb - 1]

        return df, full_stack

    def run_model(self):
        objects = btrack.utils.segmentation_to_objects(
            segmentation=self.seg_imgs.astype(int),
            intensity_image=self.raw_imgs,
            assign_class_ID=True,
            properties=tuple(self.features),
        )
        config_file = Path(bayesian_tracking.__file__).parent.joinpath("btrack_conf.json")

        cum_sum_cells = np.sum([len(np.unique(s)) - 1 for s in self.seg_imgs])
        max_cells_total = len(self.seg_imgs) * 1_000

        with btrack.BayesianTracker() as tracker:
            if cum_sum_cells < max_cells_total:
                tracker.update_method = BayesianUpdates.EXACT
            else:
                tracker.update_method = BayesianUpdates.APPROXIMATE
                tracker.max_search_radius = 256

            tracker.configure(config_file)

            tracker.features = list(self.features)
            tracker.tracking_updates = ["VISUAL", "MOTION"]

            tracker.append(objects)
            tracker.track(step_size=100)
            tracker.optimize()

            return tracker.tracks


def drop_small_objects(mask: np.ndarray, min_area: int = MIN_CELL_AREA) -> np.ndarray:
    '''
    Remove labelled objects below min_area px from a labelled mask.

    The segmentation produces a long tail of very small fragments (slivers
    shaved off a cell edge, specks of debris). They are a small fraction of the
    pixels but a large fraction of the *objects*, and each one is a link the
    tracker has to explain: they produce spurious one- and two-frame tracks and
    can capture a link that belonged to a real cell.
    '''
    if min_area <= 0:
        return mask
    cleaned = mask.copy()
    for region in regionprops(mask.astype(int)):
        if region.area < min_area:
            cleaned[mask == region.label] = 0
    return cleaned


def adapt_output_to_midap(root_path: str, frame_ini: int, frame_end: int,
                          shifts=None, min_area: int = MIN_CELL_AREA) -> None:
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

        list_shifted_im, dict_shifts = correct_shift(tiffile, maskfile, shifts=shifts)
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
            arr = drop_small_objects(arr, min_area)

            tiffwrite(trap_path + '/midap/seg_im/im_frame' + f"{n:03d}" + '_seg.tif', arr, 'XY')

        mask.close()

        im = io.imread(tiffile)
        for n in range(frame_ini, frame_end + 1):
            im_for_png = Image.fromarray(im[n])
            im_for_png.save(trap_path + "/midap/cut_im/im_frame" + f"{n:03d}" + "_cut.png")

def resolve_fluor_pattern(movie, fluor_offset, fluor_step, no_fluor):
    """Settle which frames hold fluorescence images.

    Uses whatever the user gave on the command line and works out the rest from
    the movie itself, so -fo/-fs no longer have to be counted by hand - which is
    easy to get wrong, and wrong values quietly corrupt every later step by
    overwriting the wrong frames.

    Returns (offset, step, frames).
    """
    detected = fluorescence.detect_fluor_frames(movie)
    det_offset, det_step = fluorescence.fluor_offset_and_step(detected, len(movie))

    if detected:
        regular = detected == fluorescence.list_fluor_frames(
            len(movie), det_offset, det_step or len(movie))
        print(f"Found {len(detected)} fluorescence frames "
              f"(offset {det_offset}, step {det_step})")
        if not regular:
            print("  WARNING: they are not evenly spaced, so a single offset and "
                  "step cannot describe them. Detected at frames: "
                  f"{detected}")
    else:
        print("Found no fluorescence frames; treating the movie as bright-field "
              "throughout")

    # anything given explicitly wins, but say so when it disagrees
    offset = fluor_offset if fluor_offset is not None else det_offset
    step = fluor_step if fluor_step is not None else det_step

    if (fluor_offset, fluor_step) not in ((None, None), (det_offset, det_step)):
        described = f"the detected ({det_offset}, {det_step})" if detected else "none detected"
        print(f"  NOTE: using the offset/step you gave ({offset}, {step}) rather "
              f"than {described}")

    if offset is None or step is None:
        if not no_fluor and detected:
            # one fluorescence frame, or none of the pair supplied: fall back to
            # the frames themselves rather than inventing a step
            return (det_offset, len(movie), detected)
        return (0, len(movie) + 1, [])

    return offset, step, fluorescence.list_fluor_frames(len(movie), offset, step)


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
        help=("How many frames before the first fluorescence frame? Worked out"
              " from the movie if omitted."),
        type=int,
        default=None
    )

    parser.add_argument(
        "-fs",
        "--fluor_step",
        help=("How many frames in between fluorescence frames? Worked out from"
              " the movie if omitted."),
        type=int,
        default=None
    )

    parser.add_argument(
        "-tx",
        "--trap_x",
        help=("width of the center of the trap, in px. Measured from the movie if"
              " omitted."),
        type=int,
        default=None
    )

    parser.add_argument(
        "-ty",
        "--trap_y",
        help=("height of the center of the trap, in px. Measured from the movie if"
              " omitted."),
        type=int,
        default=None
    )

    parser.add_argument(
        "-ma",
        "--min_cell_area",
        help=("Segmented objects smaller than this (px) are discarded as debris"
              " before tracking. Use 0 to keep everything."),
        type=int,
        default=MIN_CELL_AREA
    )

    parser.add_argument(
        "-st",
        "--single_trap",
        help=("The movie is already a crop of one trap, so do not go looking for"
              " traps in it. Skips template matching, and -t/-at are then unused."),
        default=False,
        action='store_true'
    )

    parser.add_argument(
        "-at",
        "--auto_template",
        help=("Build the trap template from the movie instead of using -t or"
              " cropping one by hand. Off by default: it needs no work from you"
              " and cannot contain a cell or a speck of dirt, but it does not yet"
              " centre on the trap as accurately as a hand-cropped template, and"
              " on the 25 frame example that costs mother traces (median 13"
              " against 18)."),
        default=False,
        action='store_true'
    )

    parser.add_argument(
        "-cc",
        "--cell_correspondence",
        help=("Run YeaZ's frame-to-frame label matching during segmentation. Off"
              " by default: nothing downstream uses cross-frame label identity,"
              " and it costs more time than the neural network itself."),
        default=False,
        action='store_true'
    )

    parser.add_argument(
        "-t",
        "--template",
        help=("Path to a template of an empty trap. Built from the movie if"
              " omitted."),
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

    parser.add_argument(
        "--match_threshold",
        help=("How well a patch must match the trap template to count as a trap. "
              "A template cut from another position of the same experiment scores "
              "lower everywhere, and at the old fixed 0.7 that cost half the "
              "annotated cells on 2023-07-25 Pos22 - their traps were never "
              "detected. About 0.5 recovers them. Lower it when borrowing a "
              "template, leave it when the template came from this position."),
        type=float,
        default=0.7
    )

    parser.add_argument(
        "--rescale",
        help=("Whether to stretch each image to its own range before the 8-bit "
              "conversion the trap detection works on. 'ask' warns and prompts "
              "only when the movie uses little of the 16-bit range, which is "
              "when it matters; 'always' and 'never' skip the question. A "
              "non-interactive run cannot be prompted and rescales, so pass "
              "'never' to hold a batch to the old behaviour."),
        choices=('ask', 'always', 'never'),
        default='ask'
    )

    parser.add_argument(
        "-thr",
        "--thr_val",
        help=("Probability above which a pixel counts as cell. Was fixed at 0.9."
              " The weights retrained on the 2023 movies predict lower values"
              " than the shipped ones, so they want a lower threshold."),
        type=float,
        default=0.9
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

    im = io.imread(path_to_full_movie)
    Nmax, Ndim1, Ndim2 = im.shape

    fluor_offset, fluor_step, fluor_frames = resolve_fluor_pattern(
        im, args.fluor_offset, args.fluor_step, args.no_fluor)
    del im
    frame_ini = 0
    frame_end = Nmax - 1

    trap_center_size = None
    trap_center_offset = (0.0, 0.0)
    if args.trap_y is not None and args.trap_x is not None:
        trap_center_size = [args.trap_y, args.trap_x]
    folder_init_movie = Path(path_to_full_movie)
    folder_init_movie = str(folder_init_movie.parent)

    ###################################
    # step 1 : do the segmentation of the full movie [code from launch_NN_commandline]
    # Including the fluor frames. Ignore fluor segmentation and separate the fluor
    # from non-fluor frames in a post-processing step
    ###################################

    path_to_no_fluor_movie = './output/' + filename + '_no_fluor.tif'
    path_to_no_fluor_mask = './output/' + filename + "_no_fluor.h5"

    if not args.no_segmentation:
        print('\n')
        print("Overwriting the fluorescence frames before segmenting")

        # The fluorescence frames have to go *before* the network sees them.
        # Segmenting them produces nonsense (on the example movie, 4 objects on
        # one fluorescence frame and 0 on another, against ~130 real cells), and
        # that nonsense then becomes the reference frame that YeaZ matches the
        # next bright-field frame against. Replacing each fluorescence frame
        # with the bright-field frame before it keeps the label correspondence
        # chain on real data from end to end, and saves one network pass per
        # fluorescence frame.
        fluorescence.overwrite_fluor_frames(path_to_full_movie,
                                            path_to_no_fluor_movie,
                                            fluor_frames)
        print(f"  {len(fluor_frames)} fluorescence frames replaced")

        print('Running the segmentation :')
        reader = nd.Reader("", path_to_no_fluor_mask, path_to_no_fluor_movie)
        LaunchInstanceSegmentation(reader, image_type=None, fov_indices=[0], time_value1=frame_ini,
                                   time_value2=frame_end,
                                   thr_val=args.thr_val,
                                   min_seed_dist=5, path_to_weights=path_to_weights, device='cuda',
                                   duplicate_frames=fluor_frames,
                                   use_correspondence=args.cell_correspondence)

        print('\n Segmentation step done')

        # the rest of the pipeline still refers to the mask by its original
        # name, and the two are now identical
        shutil.copyfile(path_to_no_fluor_mask, path_to_first_mask)
    print('\n')

    # ###################################
    # # step 2 : separate the traps from the movie
    # ###################################

    output_folder = './output'

    if args.single_trap:
        print("Treating the movie as a single trap; not looking for traps in it")
        path_to_splitted_traps = template_matching.single_trap_split(
            path_to_no_fluor_movie, path_to_no_fluor_mask, output_folder)
        path_to_template = None
    else:
        path_to_splitted_traps = None

    background = None
    generated_template = False
    if not args.single_trap and args.auto_template and not path_to_template:
        print("Building a template of an empty trap from the movie")
        # deliberately not inside a timestamped run folder: template_matching
        # reads the template's parent directory name and, if that names an
        # existing output folder, takes it as a request to reuse that run's
        # split rather than to split afresh
        path_to_template, _bg_path, background = trap_detection.make_template(
            path_to_no_fluor_movie, path_to_no_fluor_mask,
            Path(output_folder) / filename)
        generated_template = True

    if not args.single_trap:
        path_to_splitted_traps = template_matching.main(
            path_to_no_fluor_movie,
            path_to_no_fluor_mask,
            path_to_template,
            output_folder,
            match_image=background,
            rescale=args.rescale,
            threshold=args.match_threshold
        )

    if generated_template:
        # keep the template and the background with the run they produced
        for made in (path_to_template, _bg_path):
            shutil.copyfile(made, str(path_to_splitted_traps) + '/' + os.path.basename(made))

    print("\n")

    ###################################
    # step 3 : adapt the output format for midap
    ###################################

    if not args.no_format_midap:
        print("Estimating the drift of the field of view")
        global_shifts = estimate_global_drift(path_to_no_fluor_movie)
        adapt_output_to_midap(path_to_splitted_traps, frame_ini, frame_end,
                              shifts=global_shifts, min_area=args.min_cell_area)

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
                bst = TrackingWithFeatures(imgs=imgs, segs=segs, model_weights=None,
                                           duplicate_frames=fluor_frames)
                data_file, csv_file = bst.track_all_frames(trap_path + '/midap')
                print("Tracking saved in ", data_file, csv_file)
            except EmptyTrap as e:
                print(f"  Trap {trap_path:s} is empty ({e}), nothing to track")
                continue
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

        if trap_center_size is None:
            print("Measuring the trap centre from the cells the traps are holding")
            measured = posttreatment.measure_trap_centre(sorted(list_traps))
            if measured is None:
                raise RuntimeError(
                    "No trap held a cell for long enough to measure the trap "
                    "centre. Give it with -tx/-ty.")
            trap_center_size, trap_center_offset = measured
            print(f"  trap centre {trap_center_size[1]} x {trap_center_size[0]} px "
                  "(width x height)")

        if path_to_template:
            check_image = posttreatment.save_trap_centre_check(
                path_to_template, trap_center_size,
                str(path_to_splitted_traps) + '/trap_centre_check.png',
                trap_center_offset=trap_center_offset)
            print(f"  sanity check written to {check_image}")

        for i, trap_path in enumerate(list_traps):
            print(f'Finding mother cell for {trap_path:s}')
            try:
                posttreatment.id_the_mother(trap_path, trap_center_size,
                                            trap_center_offset=trap_center_offset)
            except FileNotFoundError:
                print(f"File not found for {trap_path:s}. Is the trap empty for all frames?")
                pass

    ###################################
    # step 6 : Get the fluorescence values of each mother cell
    ###################################

    if not args.no_fluor:
        list_traps = glob.glob(str(path_to_splitted_traps) + '/split_data/*')
        for i, trap_path in enumerate(list_traps):
            print(f'Processing fluorescence data from {trap_path:s}')

            try:
                fluorescence.get_fluorescence_data(path_to_full_movie,
                                                   path_to_first_mask,
                                                   trap_path,
                                                   Nmax, fluor_offset, fluor_step)
                posttreatment.generate_summary_plots(trap_path)
            except FileNotFoundError:
                print(f"File not found for {trap_path:s}")
                pass

    posttreatment.summary_csv(path_to_splitted_traps)
    posttreatment.check_segmentation_and_tracking(path_to_full_movie,
                                                  path_to_first_mask,
                                                  path_to_splitted_traps)

if __name__ == '__main__':
    main()
