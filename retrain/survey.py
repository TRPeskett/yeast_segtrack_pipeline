"""Measure how well the current model segments each position, worst first.

Retraining is worth doing where the model is actually failing, and on these
movies that is not uniform. The first diagnosis - that the weights do not
transfer to the 2023 imaging condition - was made on frame 0 of a single
position, and does not generalise: `2023-05-23 Pos6` segments at the same
quality as the 2021 movie the pipeline was developed on, from its first frame,
while `2023-07-20 Pos0` finds specks at a median area of 27 px in every frame.

So "the 2023 condition" is not one condition. This ranks the positions by how
badly they segment, so that correcting effort goes where there is something to
fix rather than to whichever position happens to sort first.

There is no ground truth to score against, so quality is judged by what a cell
cannot be. In these movies a yeast cell is 200-400 px: an object under 120 px is
a fragment, one over 600 px is trap structure or two cells merged, and an object
lying on the static structure of the movie is trap painted as a cell.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main  # noqa: E402
from skimage.measure import regionprops  # noqa: E402
from make_crops import DATA, movies_for, static_structure  # noqa: E402

FRAGMENT = 120
MERGED = 600


def measure(segmenter, movie, trap_mask, frames):
    """Object statistics for a few frames of one movie."""
    rows = []
    with tifffile.TiffFile(movie) as handle:
        n_frames = len(handle.pages)
        for frame_nb in frames:
            if frame_nb >= n_frames:
                continue
            labelled = segmenter.segment_frame(handle.pages[frame_nb].asarray(),
                                               thr_val=0.9, min_seed_dist=5)
            regions = regionprops(labelled.astype(np.int32))
            if not regions:
                rows.append(dict(frame=frame_nb, objects=0, median_area=0,
                                 fragments=0.0, merged=0.0, on_trap=0.0))
                continue
            areas = np.array([region.area for region in regions])
            on_trap = np.mean([trap_mask[tuple(region.coords.T)].mean() > 0.5
                               for region in regions])
            rows.append(dict(
                frame=frame_nb,
                objects=len(areas),
                median_area=float(np.median(areas)),
                fragments=float((areas < FRAGMENT).mean()),
                merged=float((areas > MERGED).mean()),
                on_trap=float(on_trap)))
    return rows


def survey(weights, n_frames=4, output='retrain/survey.csv'):
    segmenter = main.Segmenter(weights, device='cuda')
    records = []
    for experiment in sorted(os.listdir(DATA)):
        directory = os.path.join(DATA, experiment)
        if not os.path.isdir(directory):
            continue
        movies = movies_for(directory)
        for position in sorted(movies):
            movie = movies[position]
            with tifffile.TiffFile(movie) as handle:
                total = len(handle.pages)
            # first frame included on purpose: it is the one the original
            # diagnosis was made on, and on some positions it is the worst
            frames = sorted(set(np.linspace(0, total - 1, n_frames).astype(int)))
            trap_mask = static_structure(movie)
            for row in measure(segmenter, movie, trap_mask, frames):
                records.append({'experiment': experiment, 'position': position,
                                **row})
            print(f'  {experiment[:26]:26s} Pos{position:<3}', flush=True)

    frame = pd.DataFrame(records)
    frame.to_csv(output, index=False)

    by_position = frame.groupby(['experiment', 'position']).agg(
        objects=('objects', 'median'),
        median_area=('median_area', 'median'),
        fragments=('fragments', 'mean'),
        on_trap=('on_trap', 'mean')).reset_index()
    # the same two failures the crop ordering uses, so the two agree
    by_position['badness'] = by_position.fragments + by_position.on_trap
    by_position = by_position.sort_values('badness', ascending=False)

    print(f'\n{"experiment":28s} {"pos":>4s} {"obj":>5s} {"median":>7s} '
          f'{"frag":>6s} {"trap":>6s} {"badness":>8s}')
    print('-' * 70)
    for _, row in by_position.iterrows():
        print(f'{row.experiment[:28]:28s} {row.position:4d} {row.objects:5.0f} '
              f'{row.median_area:7.0f} {row.fragments:5.0%} {row.on_trap:5.0%} '
              f'{row.badness:8.2f}')
    print(f'\nfull per-frame table -> {output}')
    return by_position


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--weights', default=os.getenv(
        'WEIGHTS_YEAST', 'weights/weights_budding_BF_multilab_0_1'))
    parser.add_argument('--frames', type=int, default=4,
                        help='frames sampled per position')
    parser.add_argument('--output', default='retrain/survey.csv')
    args = parser.parse_args()
    survey(args.weights, args.frames, args.output)
