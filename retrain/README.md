# Retraining the segmentation model for the 2023 movies

The shipped YeaZ weights do not transfer to the imaging condition the 2023
ageing movies were shot in. On `2023-05-23 Pos3` frame 0 they give 379 objects at
a median area of 124 px, against 130 objects at 304 px on the movie the pipeline
was developed on: cells are fragmented, and in some configurations the trap arms
are segmented as cells.

Magnification, intensity range, contrast polarity, static-structure removal and
the phase-contrast weights were all tested and ruled out. The model needs
examples from this condition.

## The loop

Everything below needs the pipeline's conda environment. Without it you get
`command not found: python`, because `python` is not on the PATH otherwise:

```bash
conda activate segtrack
```

(or use `~/miniconda3/envs/segtrack/bin/python` in place of `python` throughout).

```bash
# 1. cut crops out of the movies and pre-segment them with the current model
python retrain/make_crops.py --name round1

# 2. correct them by hand
python retrain/annotate.py --round retrain/rounds/round1

# 3. fine-tune on whatever you corrected
python retrain/finetune.py --output weights/weights_2023_finetuned

# 4. point the pipeline at the new weights and see whether it moved
#    (edit WEIGHTS_YEAST in .env, then re-run on a position)
```

Then cut a `round2` with the improved model, which should need less correcting,
and repeat. Progress is remembered per round, so stopping half way through loses
nothing.

## Correcting

**You are painting cell against not-cell.** The network predicts one channel and
`segment()` separates touching cells afterwards by watershed, so the numbering on
the labels does not matter - training binarises them. Do not spend time giving
each cell its own colour or number.

**But leave a gap between touching cells.** This is the one thing about the
numbering that does matter, indirectly: at inference, two cells are split only if
the painted region is waisted enough for the distance transform to show two
peaks. Two cells painted as a single smooth blob come back out as one cell
whatever label values were used. Paint each outline properly and let them nearly,
but not quite, meet.

Two things worth getting right, because they are what the model is failing at:

- **the trap is not a cell.** Leave the trap arms unpainted, especially where
  the model has painted them.
- **a fragmented cell is one cell.** Where the model has split one cell into
  pieces, it should end up as a single painted region.

Keys, while the napari window has focus:

| key | |
|---|---|
| `d` | mark this crop finished and jump to the next unfinished one |
| `c` | clear this crop, to draw from scratch |
| `r` | put the model's original guess back |
| `s` | save now (it also saves on close) |
| `n` | skip to the next unfinished crop |

napari's own shortcuts, which do the actual drawing:

| key | |
|---|---|
| `2` / `1` / `3` | paint / erase / fill |
| `4` / `5` | colour picker (pick up a cell's label) / pan and zoom |
| `[` / `]` | smaller / bigger brush |
| `M`, `=`, `-` | new label, next label, previous label - rarely needed here |
| `Ctrl+Z` | undo |

`c` is there for crops where the guess is so bad that fixing it is slower than
starting clean.

## Cells cut off at the edge of a crop

**Paint what you can see, and do not worry about them.** Do not erase them.

Those pixels really are cell, so erasing would teach the model they are
background, which is worse than leaving them. But a sliver at the very edge is
often genuinely undecidable, and that should not be your problem - so an 8 px
border of every crop is **excluded from the training loss** (`--border`,
`finetune.py`). Whatever is or is not painted out there teaches the model
nothing either way.

The crop boundary is an artefact of how this training data was made. At
inference the model sees the whole 1024x1024 frame, where those cells are
whole.

Note `r` above shadows napari's add-rectangle shortcut. There is no Shapes layer
in this tool, so nothing is lost by it.

## Notes

- Crops are 176x176, centred on traps. Where the annotator's Fiji ROIs exist
  those are used; otherwise traps are found from the periodicity of the array,
  which works on these movies even though the segmentation does not.
- Validation is split **by microscope position**, never at random: crops from
  one position and nearby frames show the same cells in the same trap, so a
  random split would put near-duplicates on both sides and report a score that
  means nothing.
- Training starts from the existing bright-field weights at a low learning rate.
  A few dozen crops is far too little to train a U-Net of this size from scratch.
- The first round samples all four experiments and spreads across each movie, so
  the model sees sparse early frames and crowded late ones rather than learning
  one situation well.
