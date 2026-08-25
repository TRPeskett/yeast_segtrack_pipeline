"""Correct trap-crop segmentations in napari, to build training data.

midap already has a corrector, but it works one frame at a time behind a
matplotlib button: open napari, fix a frame, close it, click Next, open it again.
That is fine for spot-fixing a run and far too slow for producing a training set.

Here the whole round is loaded as a single stack, so the crops are a slider away
from each other, and progress is remembered - so an hour spent now is not lost if
the rest happens next week.

What matters for training is only whether a pixel is cell or not: `segment()`
separates touching cells afterwards by watershed, so there is no need to give
each cell its own number. Paint the outlines correctly and ignore the colours.

Keys
  d   mark this crop finished, and go to the next unfinished one
  c   clear this crop, to draw it from scratch
  r   put the model's original guess back
  s   save now

Saving is automatic: whenever you move between crops, whenever you press d, and
when you close the window. s is there for peace of mind.
  n   jump to the next unfinished crop
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import tifffile


def load_round(stem):
    images = tifffile.imread(stem + '_images.tif')
    labels = tifffile.imread(stem + '_labels.tif')
    index = pd.read_csv(stem + '_index.csv')
    if 'corrected' not in index.columns:
        index['corrected'] = False
    return images, labels, index


def save_round(stem, labels, index):
    tifffile.imwrite(stem + '_labels.tif', labels.astype(np.uint16))
    index.to_csv(stem + '_index.csv', index=False)


def describe(index, i):
    row = index.iloc[i]
    done = index['corrected'].sum()
    return (f"crop {i + 1}/{len(index)}   {row.experiment[:24]}  Pos{row.position} "
            f"trap{row.trap} frame{row.frame}   "
            f"[{'DONE' if row.corrected else 'not corrected'}]   "
            f"{done}/{len(index)} finished")


def run(stem):
    import napari

    images, labels, index = load_round(stem)
    original = labels.copy()

    viewer = napari.Viewer(title=f'correcting {os.path.basename(stem)}')
    viewer.add_image(images, name='image', contrast_limits=(
        float(np.percentile(images, 1)), float(np.percentile(images, 99))))
    label_layer = viewer.add_labels(labels, name='cells', opacity=0.45)

    # ready to paint straight away, with a brush about the width of a bud
    label_layer.mode = 'paint'
    label_layer.brush_size = 6
    label_layer.selected_label = 1

    viewer.text_overlay.visible = True
    viewer.text_overlay.font_size = 11

    def current():
        return int(viewer.dims.current_step[0])

    def refresh():
        viewer.text_overlay.text = describe(index, current())

    def go_to(i):
        step = list(viewer.dims.current_step)
        step[0] = int(i)
        viewer.dims.current_step = tuple(step)
        refresh()

    def next_unfinished(after):
        pending = index.index[~index['corrected']]
        later = [i for i in pending if i > after]
        if later:
            return later[0]
        return pending[0] if len(pending) else None

    @viewer.bind_key('d', overwrite=True)
    def mark_done(_v):
        i = current()
        index.loc[i, 'corrected'] = True
        save_round(stem, label_layer.data, index)
        nxt = next_unfinished(i)
        if nxt is None:
            viewer.text_overlay.text = 'every crop is corrected - nothing left'
            return
        go_to(nxt)

    @viewer.bind_key('c', overwrite=True)
    def clear(_v):
        label_layer.data[current()] = 0
        label_layer.refresh()
        refresh()

    @viewer.bind_key('r', overwrite=True)
    def restore(_v):
        label_layer.data[current()] = original[current()]
        label_layer.refresh()
        refresh()

    @viewer.bind_key('s', overwrite=True)
    def save(_v):
        save_round(stem, label_layer.data, index)
        viewer.text_overlay.text = describe(index, current()) + '   (saved)'

    @viewer.bind_key('n', overwrite=True)
    def skip(_v):
        nxt = next_unfinished(current())
        if nxt is not None:
            go_to(nxt)

    def on_move(_event):
        """Save on the way out of a crop, so nothing depends on a clean exit.

        Closing the window saves too, but a force-quit or a crash would not, and
        an hour of careful correcting is too much to lose to that. Writing the
        stack takes a few tens of milliseconds, which is unnoticeable next to
        moving between crops.
        """
        save_round(stem, label_layer.data, index)
        refresh()

    viewer.dims.events.current_step.connect(on_move)

    start = next_unfinished(-1)
    if start is not None:
        go_to(start)
    refresh()

    print(__doc__)
    napari.run()

    # napari.run returns when the window closes
    save_round(stem, label_layer.data, index)
    done = int(index['corrected'].sum())
    print(f'\nsaved: {done}/{len(index)} crops marked corrected -> {stem}_labels.tif')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--round', default='retrain/rounds/round1',
                        help='path stem of the round, without _images.tif')
    args = parser.parse_args()
    run(args.round)
