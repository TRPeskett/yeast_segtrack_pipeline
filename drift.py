"""Estimating and undoing the drift of a movie, independently of the pipeline.

The microscope stage wanders over a long acquisition, so a trap is not at the
same pixel on frame 300 as on frame 0. Everything downstream assumes it is.

This lives on its own because the job is useful on its own: it needs only numpy,
scipy and scikit-image, where importing the pipeline costs about six seconds and
drags in torch, yeaz, midap and btrack. Nothing here knows about traps, masks,
h5 files or output folders.

    import drift, tifffile
    stack  = tifffile.imread('movie.tif')
    shifts = drift.estimate_drift(stack)
    fixed  = drift.apply_drift(stack, shifts)
"""

import numpy as np
from scipy import ndimage
from skimage.registration import phase_cross_correlation

# Whole-pixel shifts by default. Sub-pixel registration would need
# upsample_factor below and an interpolating order in apply_drift, and the
# pipeline rounds to whole pixels anyway.
DEFAULT_UPSAMPLE = 1


def for_registration(frame):
    """A frame as the drift estimation should see it: full precision, no 8-bit.

    This is the real definition. `main.for_registration` is an alias to it, so
    edit it here - changing it in main.py has no effect.

    This used to go through `img_as_ubyte` on the raw uint16, a fixed divide by
    257. That is the same defect that cost the trap detection half its traps on
    the 2023 movies, and it bites here too: those movies use ~7% of the 16-bit
    range, so only 16 distinct grey levels survive the conversion, against 130
    for the 2021 movies.

    Unlike `matchTemplate`, which needs 8-bit or float32, `phase_cross_correlation`
    is happy with anything numeric - so the conversion bought nothing and threw
    away precision. Measured, it turns out to have been harmless: the drift
    estimates agree to within 1 px on both a 2021 and a 2023 movie, and the
    pipeline rounds shifts to whole pixels anyway. Phase correlation keeps only
    the phase spectrum and discards magnitude, which is exactly what a rescale
    changes, and the trap lattice is periodic enough to survive coarse
    quantisation. Removed regardless: a lossy step that buys nothing should not
    sit in front of a measurement, and a movie using even less of the range
    would eventually be left with too few levels to lock on to.
    """
    return np.asarray(frame, dtype=np.float32)


def estimate_drift(frames, reference=0, upsample=DEFAULT_UPSAMPLE, whole_pixels=True):
    """Per-frame (dy, dx) needed to bring each frame back onto the reference.

    Estimated on whatever it is given, which for the pipeline is the whole field
    of view. That matters: a single trap crop that fills up with cells has
    little fixed structure left for the correlation to lock on to, which is what
    made per-crop drift correction fail. The full frame keeps the trap array,
    which barely changes.

    One caveat worth knowing. A field of identical traps is close to periodic,
    so the correlation surface has peaks repeating at the trap spacing, and the
    estimate is only unambiguous while the true peak stays the strongest. It
    does on these movies - measured at a median of 1 px frame to frame, never
    above 4 px - but a coarser or noisier image can make a neighbouring peak win
    and displace everything by a whole trap. Estimate at full resolution; do not
    downsample first to save time.

    frames : sequence of 2D arrays, or a 3D array indexed by time
    reference : index of the frame everything is registered to
    """
    frames = list(frames)
    anchor = for_registration(frames[reference])

    shifts = []
    for frame in frames:
        shift, _error, _phase = phase_cross_correlation(
            anchor, for_registration(frame), upsample_factor=upsample)
        shifts.append(np.round(shift) if whole_pixels else shift)

    return np.array(shifts)


def apply_drift(frames, shifts, order=0, cval=0):
    """Move each frame by its shift, returning a stack of the same dtype.

    order=0 - nearest neighbour - is the default because this is also used on
    label images, where a cell is an integer id and interpolating between two of
    them invents a value belonging to no cell. On plain images order=1 is
    smoother; use it only where the result is not a label image.
    """
    frames = np.asarray(frames)
    moved = [ndimage.shift(frame, shift, order=order, mode='constant', cval=cval)
             for frame, shift in zip(frames, shifts)]
    return np.stack(moved).astype(frames.dtype)


def apply_drift_to_frame(frame, shift, order=0, cval=0):
    """One frame, for callers holding a movie open rather than in memory."""
    frame = np.asarray(frame)
    return ndimage.shift(frame, shift, order=order,
                         mode='constant', cval=cval).astype(frame.dtype)


def residual_drift(frames, reference=0):
    """Largest drift still present, for checking that a correction worked."""
    shifts = estimate_drift(frames, reference=reference, whole_pixels=False)
    return float(np.max(np.hypot(shifts[:, 0], shifts[:, 1])))
