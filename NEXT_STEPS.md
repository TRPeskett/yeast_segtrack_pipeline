# Next steps

Written up 2026-08-19, after running the pipeline on three movies: the 25-frame
`small_movie.tif`, the 364-frame
`2021-12-15_ageing_optoControls_lightOff_Pos7-1_fr1-364.tif`, and the 90-frame
`medium_movie_hard.tif`.

Tracking is now good enough that **the errors visible by eye are segmentation
errors, not tracking errors**. That is what the list below is ordered by.

---

## 1. Retrain the YeaZ segmentation model (highest value)

**Updated 2026-08-20: this is now a hard blocker, not just an improvement.** The
2023 ageing movies in `/Users/ucbtpes/yeast_division_movies` - the ones the
hand-annotated division ground truth was made on - are shot in a different
imaging condition, and the model does not transfer to it at all. Frame 0 of
`2023-05-23 Pos3` segments to 379 objects at median area 124 px, against 130
objects at median 304 px for the movie the pipeline was developed on. Over half
are fragments, and in some configurations the trap arms are segmented as cells.

Ruled out as causes, so do not retry: magnification (the trap lattice is 111x147
against 110x147, i.e. identical), intensity range (`equalize_adapthist` rescales
internally), contrast polarity (inverting makes the traps worse), static
structure removal (fixes trap contamination but not fragmentation), and the
phase-contrast weights (worse everywhere, including on the 2021 movie). See
`STATE.md` in the division-detection project for the full table.

The remaining errors trace back to the segmentation, and the model is being asked
for three things it currently does badly. All need training data.

### Crowded cells
Where several cells are pressed together the boundaries between them are missed or
put in the wrong place. Worst on `medium_movie_hard.tif`, which is also the lowest
contrast of the three movies.

Examples to draw training data from (`output/2026-08-19_16-45-33`):

| trap | up to | busiest frame |
|---|---|---|
| 27 | 8 cells | 81 |
| 8 | 7 cells | 3 |
| 10 | 7 cells | 5 |
| 36 | 7 cells | 3 |
| 5 | 6 cells | 64 |
| 22 | 6 cells | 71 |

### A different imaging condition
The 2023 movies show thin dark trap outlines and small dark cells; the movies the
pipeline was developed on have brightly embossed traps. This is the blocker
described above, and the annotated data is in exactly this condition - so the
training data and the ground truth are the same movies.

### Unusual cell morphology
Old mothers late in a long movie stop being round, and the model loses them or
splits them. This is what breaks the last stretch of several mother traces on the
364-frame movie. Noted from the manual review of `output/2026-08-19_11-58-51`:

- **trap 16** - the mother goes odd near the end of the movie; identity moves
  between the cell and its neighbours as the segmentation flickers
- **trap 21** - segmentation struggles, 89 tracks in one trap over 364 frames
- **trap 29** - same pattern late in the movie

Frames past ~300 are where to look.

### How to get the data out
`correct_segmentation` writes corrected masks in place, so a corrected trap is
already in the right form to train on:

```bash
correct_segmentation --path_img ./output/<timestamp>/split_data/<trap>/midap/cut_im --path_seg ./output/<timestamp>/split_data/<trap>/midap/seg_im
```

Note it works on *binary* masks, so corrections fix cell outlines but not which
cell is which - that is fine for segmentation training.

---

## 2. Per-trap trap centre offset

The trap centre box is measured from the movie (`posttreatment.measure_trap_centre`)
and placed at a **single offset shared by every trap**, the median across traps.
Traps whose crop is offset differently from the consensus then have their cells
fall outside the box and get no mother at all.

Two clear cases in `output/2026-08-19_16-45-33`, against a box of +-16, +-10 px:

- **trap 17**: cells sit 27 px left of the crop centre - all 158 of them classed
  outsider, no mother found
- **trap 51**: cells sit 18 px below - only 2 mother frames out of 90

`measure_trap_centre` already computes each trap's own centre before taking the
median across traps, so the per-trap numbers exist; they are currently thrown
away. Using each trap's own offset, falling back to the consensus for traps that
held nothing, would fix both cases. Contained change, good value.

---

## 3. Template centring, if `-at` is ever to become the default

`-at` builds a template from the movie and finds the traps as well as a
hand-cropped one (peak match 0.96, and one trap more on the 364-frame movie), but
it centres about 10 px off the trap waist, the crops inherit that, and on the
25-frame movie it costs mother traces (median 13 against 18). Four centrings were
tried; see the comment in `trap_detection.build_template` for which and why they
failed, so they are not repeated.

---

## Things already measured, so as not to redo them

- **btrack's config is not the problem.** Sweeping `max_lost`, the motion model
  sigmas and `accuracy` made things worse in every case. `btrack_conf.json` is
  untouched on purpose.
- **Per-frame intensity normalisation before the network is not the answer.**
  Rescaling each frame to its own 1st/99th percentile evens out the object counts
  but clips the tails that carry the cell edges: real cells shrink, some vanish,
  and full-length tracks fell from 66 to 46.
- **Renumbering the tracker's frames to real frame indices is wrong.** A
  fluorescence image is acquired alongside a bright-field one, not at a time
  point of its own, so the bright-field frames either side of it are one step
  apart. Treating the gap as two steps cost 372 -> 495 tracks over six traps.
