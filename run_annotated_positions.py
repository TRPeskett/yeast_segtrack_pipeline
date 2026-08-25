"""Run the pipeline over every annotated position, one at a time, and prune.

Each run produces about 12 GB, almost all of it h5 masks and a check movie that
nothing downstream reads. Fifteen of those would be 180 GB against 260 GB free,
so each run is pruned as soon as it finishes to the ~20 MB that feature
extraction actually needs: the per-trap tracking csv and the seg_im masks.

The trap boxes are saved to a csv first, because `posttreatment.trap_boxes`
recovers them by parsing the *filename* of a 14 MB crop that is otherwise dead
weight.

Positions are ordered so that one from each experiment runs first. Three of the
four experiments have never been through the pipeline and their templates are
borrowed from a position that is not in the annotated set, so if an experiment is
going to fail it should fail in the first hour rather than the ninth.

This script is a personal research harness rather than part of the pipeline
proper, and it will not run from a fresh clone of this repository. It imports
`divisions.labels` and `divisions.rois` from a separate `yeast_division_detection`
package, and reads hand-annotation spreadsheets and movies from a
`yeast_division_movies` archive; neither is included here. Set
YEAST_DIVISION_DETECTION and YEAST_MOVIES if you keep them somewhere other than
your home directory. Nothing else in the pipeline depends on this file.
"""

import csv
import glob
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The division-detection package lives outside this repository. Set
# YEAST_DIVISION_DETECTION if you keep it somewhere other than your home
# directory.
sys.path.insert(0, os.environ.get(
    'YEAST_DIVISION_DETECTION',
    os.path.expanduser('~/yeast_division_detection')))

from divisions.labels import load_labels          # noqa: E402
from divisions.rois import find_movies            # noqa: E402

# The movie archive these positions are drawn from. Override with YEAST_MOVIES.
DATA = os.environ.get('YEAST_MOVIES',
                      os.path.expanduser('~/yeast_division_movies'))
PIPELINE = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.expanduser('~/miniconda3/envs/segtrack/bin/python')
WEIGHTS = 'weights/weights_2023_finetuned'
LEDGER = os.path.join(PIPELINE, 'annotated_runs.csv')

# A run directory is named by timestamp, "%Y-%m-%d_%H-%M-%S". Match the shape
# rather than any particular year: globbing for one year's runs silently finds
# nothing once the year turns, which would leave `run` empty and skip both the
# trap boxes and the pruning that keeps this script inside the free disk space.
RUN_DIR = '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9]-[0-9][0-9]-[0-9][0-9]'

EXPERIMENT = {
    '2023-05-16': '2023-05-16_ageing_optoWhi3_PBmutants',
    '2023-05-23': '2023-05-23_ageing_optoWhi3_PBmutants_noLight',
    '2023-07-20': '2023-07-20_ageing_optoWhi3_Lsm4dC_Edc3dYjef_LightOn',
}
TEMPLATE = {
    '2023-05-16': 'templates/2023-05-16_ageing_optoWhi3_PBmutants_Pos0_template.tif',
    '2023-05-23': 'templates/2023-05-23_ageing_optoWhi3_PBmutants_noLight_Pos7_template.tif',
    '2023-07-20': 'templates/2023-07-20_ageing_optoWhi3_Lsm4dC_Edc3dYjef_LightOn_Pos0_template.tif',
}
# one per experiment first, so a broken template shows up early
FIRST = [('2023-05-23', 3), ('2023-07-20', 17), ('2023-05-16', 0)]


def run_dirs():
    """The pipeline's timestamped run directories, whatever the year."""
    return {path for path in glob.glob(os.path.join(PIPELINE, 'output', RUN_DIR))
            if os.path.isdir(path)}


def positions():
    """[(date, position, frames_to_run)], cut at the last annotated frame.

    Running past the annotations buys nothing and costs the worst frames of the
    movie: these traps clog late, and a clogged frame segments badly enough to
    break tracks that were fine up to then.
    """
    labels = pd.concat([
        load_labels(f'{DATA}/cell_cycle_hand_annotations/2023-05-16_optoWhi3_PBmutants_cellcycle.xlsx'),
        load_labels(f'{DATA}/cell_cycle_hand_annotations/2023-07-20_optoWhi3_Lsm4dC_Edc3dYjef_cellcycle.xlsx'),
    ], ignore_index=True)
    labels['Date'] = labels['Date'].astype(str).str[:10]

    found = []
    for (date, position), group in labels.groupby(['Date', 'Pos']):
        if date not in EXPERIMENT:
            continue                      # 2023-07-25 is already done
        movies = find_movies(os.path.join(DATA, 'movies', EXPERIMENT[date]))
        position = int(position)
        if position not in movies:
            continue
        with tifffile.TiffFile(movies[position]) as handle:
            total = len(handle.pages)
        last = int(np.nanmax([group['End_life'].max(), group['End_bud'].max()])) + 1
        found.append((date, position, min(total, last + 2), movies[position]))

    order = {key: i for i, key in enumerate(FIRST)}
    return sorted(found, key=lambda r: (order.get((r[0], r[1]), 99), r[0], r[1]))


def truncate(source, destination, n_frames):
    with tifffile.TiffFile(source) as handle, tifffile.TiffWriter(destination) as writer:
        for t in range(min(n_frames, len(handle.pages))):
            writer.write(handle.pages[t].asarray(), contiguous=True)


def save_trap_boxes(run):
    """The four numbers `posttreatment.trap_boxes` parses out of a crop's name."""
    rows = []
    for trap_dir in sorted(glob.glob(os.path.join(run, 'split_data', '*'))):
        crops = [t for t in glob.glob(trap_dir + '/*.tif')
                 if 'shift_corrected' not in os.path.basename(t)]
        if not crops:
            continue
        parts = os.path.basename(crops[0]).split('_')
        try:
            box = [int(p) for p in parts[:4]]
        except ValueError:
            continue
        rows.append([int(os.path.basename(trap_dir))] + box)

    with open(os.path.join(run, 'trap_boxes.csv'), 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['trap', 'x_min', 'y_min', 'x_max', 'y_max'])
        writer.writerows(sorted(rows))
    return len(rows)


def prune(run):
    """Keep what feature extraction reads; drop the 12 GB it does not."""
    for path in glob.glob(os.path.join(run, '**', '*.h5'), recursive=True):
        os.remove(path)
    for path in glob.glob(os.path.join(run, '**', '*.tif'), recursive=True):
        if os.sep + 'seg_im' + os.sep in path:
            continue
        os.remove(path)
    for path in glob.glob(os.path.join(run, '**', 'check_segmentation_and_tracking.tiff'),
                          recursive=True):
        os.remove(path)
    for cut in glob.glob(os.path.join(run, 'split_data', '*', 'midap', 'cut_im')):
        shutil.rmtree(cut, ignore_errors=True)


def main():
    todo = positions()
    print(f'{len(todo)} positions, {sum(r[2] for r in todo)} frames\n', flush=True)

    done = set()
    if os.path.exists(LEDGER):
        done = {(r.date, int(r.position)) for r in pd.read_csv(LEDGER).itertuples()}

    for date, position, n_frames, movie in todo:
        if (date, position) in done:
            print(f'{date} Pos{position}: already done, skipping', flush=True)
            continue

        stem = f'{date}_Pos{position}_fr1-{n_frames}'
        local = os.path.join(PIPELINE, 'input', stem + '.tif')
        print(f'=== {date} Pos{position}: {n_frames} frames ===', flush=True)

        truncate(movie, local, n_frames)
        before = run_dirs()

        environment = dict(os.environ, WEIGHTS_YEAST=WEIGHTS)
        result = subprocess.run(
            [PYTHON, 'main.py', '-i', './input/' + stem + '.tif',
             '-t', os.path.join(DATA, TEMPLATE[date]),
             '-tx', '39', '-ty', '21', '-no_fl', '-thr', '0.9',
             '--rescale', 'always', '--match_threshold', '0.5'],
            cwd=PIPELINE, env=environment,
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        made = sorted(run_dirs() - before)
        run = made[-1] if made else ''

        traps = save_trap_boxes(run) if run else 0
        if run:
            prune(run)

        with open(LEDGER, 'a', newline='') as handle:
            writer = csv.writer(handle)
            if handle.tell() == 0:
                writer.writerow(['date', 'position', 'frames', 'run', 'traps', 'exit'])
            writer.writerow([date, position, n_frames, run, traps, result.returncode])

        # the truncated copy and the pipeline's working copy are both ~600 MB
        for junk in (local, local.replace('.tif', '.h5'),
                     os.path.join(PIPELINE, 'output', stem + '_no_fluor.tif'),
                     os.path.join(PIPELINE, 'output', stem + '_no_fluor.h5')):
            if os.path.exists(junk):
                os.remove(junk)

        print(f'    -> {run}  {traps} traps  exit {result.returncode}', flush=True)

    print('\nall positions done', flush=True)


if __name__ == '__main__':
    main()
