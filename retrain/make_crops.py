"""Pull trap crops out of the movies for hand correction.

The model has to be retrained because it does not transfer to the 2023 imaging
condition, and that needs examples. A trap crop is the right unit: a handful of
cells rather than the couple of hundred in a full frame, and it contains the trap
itself, which matters because the model currently mistakes trap arms for cells.

Crops come out as two stacks - the images and the current model's guess at their
segmentation - so napari can show them as a single scrollable series. An index
csv records where each crop came from, so a corrected mask can always be traced
back to a movie, position, trap and frame.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import skimage.filters
import skimage.measure
import skimage.morphology
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main  # noqa: E402
import trap_detection  # noqa: E402

# The movie archive the crops are cut from. Override with YEAST_MOVIES.
DATA = os.environ.get('YEAST_MOVIES',
                      os.path.expanduser('~/yeast_division_movies'))
CROP = 176


def movies_for(experiment_dir):
    """{position: path} across both BF_only and concatenated_BF."""
    found = {}
    for subdir in ('BF_only', 'concatenated_BF'):
        directory = os.path.join(experiment_dir, subdir)
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            match = re.search(r'Pos(\d+)\.tif$', name)
            if match:
                found.setdefault(int(match.group(1)), os.path.join(directory, name))
    return found


def trap_boxes_for(movie_path, experiment_dir, position, limit=None):
    """Trap boxes as (centre_x, centre_y), preferring the annotator's own ROIs.

    Where a RoiSet exists it is used directly: those are the traps a person chose
    to work on, and they need no detection to be right. Otherwise the traps are
    found from the periodicity of the array, which works on these movies even
    though the segmentation does not.
    """
    roi_path = os.path.join(experiment_dir, f'RoiSet_Pos{position}.zip')
    if os.path.exists(roi_path):
        import roifile
        centres = [((r.left + r.right) / 2, (r.top + r.bottom) / 2)
                   for r in roifile.roiread(roi_path)]
    else:
        first = tifffile.imread(movie_path, key=0).astype(float)
        v1, v2 = trap_detection.lattice_vectors(first)
        height, width = first.shape
        centres = [(x, y) for x, y in
                   trap_detection._lattice_points(first.shape, (width / 2, height / 2),
                                                  v1, v2, CROP // 2 + 1)]
    return centres[:limit] if limit else centres


def cut(frame, centre, size=CROP):
    """A fixed-size crop centred on a trap, kept inside the image."""
    height, width = frame.shape
    half = size // 2
    x = int(round(min(max(centre[0], half), width - half)))
    y = int(round(min(max(centre[1], half), height - half)))
    return frame[y - half:y + half, x - half:x + half]


def static_structure(movie_path, n_sample=30):
    """Mask of everything in the movie that never moves - which is the trap.

    Cells arrive, grow and wash away; the trap is in the same pixels from the
    first frame to the last, so a temporal median holds the trap and nothing
    else. Used here only to score how wrong a guess looks: an object sitting on
    the trap is one the model should not have found at all.
    """
    with tifffile.TiffFile(movie_path) as handle:
        n_frames = len(handle.pages)
        step = max(1, n_frames // n_sample)
        stack = np.stack([handle.pages[t].asarray()
                          for t in range(0, n_frames, step)]).astype(np.float32)
    median = np.median(stack, axis=0)
    edges = skimage.filters.sobel(median / median.max())
    return skimage.morphology.binary_dilation(
        edges > np.percentile(edges, 97), skimage.morphology.disk(3))


def suspicion(guess, trap_crop):
    """How wrong the model's guess for one crop looks, 0 to 2. Higher is worse.

    Both of the ways this model fails on these movies are visible without asking
    a person: it paints the trap arms as cells, and it breaks single cells into
    fragments. Neither needs ground truth to notice, so the crops can be ordered
    by how much there is to fix before anyone opens napari.

    This matters because correcting is the expensive step. A crop the model
    already gets right teaches it nothing, and the first round sampled frames
    25 to 340 - missing the start of the movie, where 31% of objects are painted
    onto the trap against 4% by frame 50.
    """
    regions = skimage.measure.regionprops(guess.astype(np.int32))
    if not regions:
        return 0.0
    on_trap = np.mean([trap_crop[tuple(region.coords.T)].mean() > 0.5
                       for region in regions])
    fragments = np.mean([region.area < 120 for region in regions])
    return float(on_trap + fragments)


def build_round(name, plan, weights, output_dir='retrain/rounds'):
    """Extract every crop in `plan` and pre-segment it with the current model.

    plan : [(experiment_dir, position, [frames], n_traps), ...]
    """
    segmenter = main.Segmenter(weights, device='cuda')

    images, labels, index = [], [], []
    for experiment_dir, position, frames, n_traps in plan:
        movies = movies_for(experiment_dir)
        if position not in movies:
            print(f'  no movie for {os.path.basename(experiment_dir)} Pos{position}')
            continue
        movie = movies[position]
        centres = trap_boxes_for(movie, experiment_dir, position, limit=n_traps)
        trap_mask = static_structure(movie)

        with tifffile.TiffFile(movie) as handle:
            n_frames = len(handle.pages)
            for frame_nb in frames:
                if frame_nb >= n_frames:
                    continue
                full = handle.pages[frame_nb].asarray()
                for trap, centre in enumerate(centres):
                    crop = cut(full, centre)
                    if crop.shape != (CROP, CROP):
                        continue
                    guess = segmenter.segment_frame(crop, thr_val=0.9, min_seed_dist=5)
                    images.append(crop)
                    labels.append(guess.astype(np.uint16))
                    index.append({
                        'experiment': os.path.basename(experiment_dir),
                        'position': position,
                        'trap': trap,
                        'frame': frame_nb,
                        'movie': os.path.relpath(movie, DATA),
                        'centre_x': centre[0],
                        'centre_y': centre[1],
                        'suspicion': suspicion(guess, cut(trap_mask, centre)),
                    })
        print(f'  {os.path.basename(experiment_dir)[:28]:28s} Pos{position:<3} '
              f'{len(frames)} frames x {len(centres)} traps')

    # worst first, so that an hour of correcting is spent where the model is
    # wrong rather than confirming crops it already gets right
    order = np.argsort([-row['suspicion'] for row in index])
    images = [images[i] for i in order]
    labels = [labels[i] for i in order]
    index = [index[i] for i in order]

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.join(output_dir, name)
    tifffile.imwrite(stem + '_images.tif', np.array(images))
    tifffile.imwrite(stem + '_labels.tif', np.array(labels))
    frame = pd.DataFrame(index)
    frame['corrected'] = False
    frame.to_csv(stem + '_index.csv', index=False)

    print(f'\n  {len(images)} crops of {CROP}x{CROP} -> {stem}_images.tif')
    print(f'  suspicion {frame.suspicion.max():.2f} (worst) to '
          f'{frame.suspicion.min():.2f} (best); correct them in the order given')
    return stem


def positions_from_survey(survey_csv, worst=2, control=1):
    """{experiment: [positions]} - the positions `survey.py` found worst, plus one.

    Which position a crop comes from matters more than which frame. Taking the
    first two positions of each experiment, as this used to, picks whichever
    sorts first: on the survey those came out ranked 13, 31, 18, 32, 4, 38, 35
    and 23 of 39, so five of the eight were in the better half of the data and
    only one was among the genuinely broken.

    The `control` position is the *best* one in each experiment, and it is there
    on purpose. Fine-tuning only on the failures is how a model gets worse at
    what it already did well, and the held-out score cannot detect that if every
    position in the round is a bad one.
    """
    frame = pd.read_csv(survey_csv)
    ranked = frame.groupby(['experiment', 'position']).agg(
        fragments=('fragments', 'mean'), on_trap=('on_trap', 'mean')).reset_index()
    ranked['badness'] = ranked.fragments + ranked.on_trap

    chosen = {}
    for experiment, group in ranked.groupby('experiment'):
        group = group.sort_values('badness', ascending=False)
        picks = list(group.position[:worst])
        picks += [p for p in list(group.position)[::-1][:control] if p not in picks]
        chosen[experiment] = picks
    return chosen


def default_plan(frames_per_position=6, traps_per_frame=2, early=2, survey=None):
    """A spread over all four experiments, several positions, and the whole movie.

    Deliberately not all from one place: the model has to cope with sparse early
    frames and crowded late ones, and with every experiment, so the first round of
    corrections should sample all of that rather than teach it one situation well.

    `early` of the frames come from the first twenty of the movie. The original
    0.1-0.9 spread skipped those to avoid the ragged ends of an acquisition, but
    on these movies the start is not ragged - it is where the model actually
    fails. Segmenting `2023-05-23 Pos3` frame by frame: 580 objects at a median
    area of 100 px at frame 0, against 210 at 216 px by frame 50 and 304 at
    297 px by frame 150, which is what the 2021 movie gives. 31% of the objects
    at frame 0 sit on the trap, against 4% by frame 50. Correcting only frames
    25 and later, as the first round did, shows the model almost nothing it does
    not already do correctly.

    `survey` is the csv from `survey.py`; given one, the positions are chosen by
    measured segmentation quality rather than by sort order.
    """
    wanted = positions_from_survey(survey) if survey else None

    plan = []
    for experiment in sorted(os.listdir(DATA)):
        directory = os.path.join(DATA, experiment)
        if not os.path.isdir(directory):
            continue
        movies = movies_for(directory)
        if wanted is not None:
            positions = [p for p in wanted.get(experiment, []) if p in movies]
        else:
            positions = sorted(movies)[:2]
        for position in positions:
            with tifffile.TiffFile(movies[position]) as handle:
                n_frames = len(handle.pages)
            frames = list(np.linspace(0, 18, early).astype(int))
            frames += list((np.linspace(0.1, 0.9, frames_per_position - early)
                            * n_frames).astype(int))
            plan.append((directory, position, sorted(set(frames)), traps_per_frame))
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='round1')
    parser.add_argument('--weights', default=os.getenv(
        'WEIGHTS_YEAST', 'weights/weights_budding_BF_multilab_0_1'))
    parser.add_argument('--frames', type=int, default=6,
                        help='frames sampled per position')
    parser.add_argument('--traps', type=int, default=2,
                        help='traps taken per frame')
    parser.add_argument('--early', type=int, default=2,
                        help='how many of those frames come from the first '
                             'twenty, where the model is worst')
    parser.add_argument('--survey', default='retrain/survey.csv',
                        help='csv from survey.py, used to pick the positions '
                             'that segment worst; pass "" to take the first two '
                             'positions of each experiment instead')
    args = parser.parse_args()

    survey = args.survey if args.survey and os.path.exists(args.survey) else None
    if args.survey and not survey:
        print(f'  no survey at {args.survey}; falling back to the first two '
              'positions of each experiment. Run retrain/survey.py first.')

    build_round(args.name, default_plan(args.frames, args.traps, args.early,
                                        survey), args.weights)
