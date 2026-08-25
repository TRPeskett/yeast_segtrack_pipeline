"""Regression tests for working out which frames of a movie hold fluorescence.

These numbers used to be given by hand with -fo/-fs, and getting them wrong is
quiet and damaging: the wrong frames are overwritten before segmentation and
every later step then works on the wrong data. Now that they are inferred, the
inference needs to stay honest - hence these tests.

Run them with `pytest tests` from the repository root, or directly with
`python tests/test_fluor_detection.py`.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fluorescence  # noqa: E402

EXAMPLE_MOVIE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'input', 'small_movie.tif')

# the example movie ships with fluorescence at these frames
EXAMPLE_FLUOR_FRAMES = [9, 15, 21]


@pytest.fixture(scope='module')
def example_frames():
    """Real bright-field and fluorescence frames taken from the example movie."""
    if not os.path.exists(EXAMPLE_MOVIE):
        pytest.skip(f'example movie not found at {EXAMPLE_MOVIE}')

    import tifffile
    movie = tifffile.imread(EXAMPLE_MOVIE)

    bright = [movie[i] for i in range(len(movie)) if i not in EXAMPLE_FLUOR_FRAMES]
    fluor = [movie[i] for i in EXAMPLE_FLUOR_FRAMES]
    return bright, fluor


def build_movie(example_frames, n_frames, offset, step):
    """A movie of n_frames with fluorescence every `step` frames from `offset`.

    Built from real frames rather than synthetic ones, so the test exercises the
    actual intensity distributions the detection has to separate.
    """
    bright, fluor = example_frames
    wanted = sorted(range(offset, n_frames, step)) if step else []

    movie, b, f = [], 0, 0
    for i in range(n_frames):
        if i in wanted:
            movie.append(fluor[f % len(fluor)])
            f += 1
        else:
            movie.append(bright[b % len(bright)])
            b += 1
    return np.array(movie), wanted


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------

def test_detects_the_example_movie():
    """The shipped example, end to end, at its documented -fo 9 -fs 6."""
    if not os.path.exists(EXAMPLE_MOVIE):
        pytest.skip('example movie not found')

    import tifffile
    movie = tifffile.imread(EXAMPLE_MOVIE)

    frames = fluorescence.detect_fluor_frames(movie)
    assert frames == EXAMPLE_FLUOR_FRAMES
    assert fluorescence.fluor_offset_and_step(frames, len(movie)) == (9, 6)


@pytest.mark.parametrize('offset, step', [
    (9, 6), (0, 4), (3, 10), (1, 2), (5, 3), (2, 7), (0, 2), (7, 8),
])
def test_detects_any_regular_spacing(example_frames, offset, step):
    movie, wanted = build_movie(example_frames, 40, offset, step)

    frames = fluorescence.detect_fluor_frames(movie)

    assert frames == wanted
    assert fluorescence.fluor_offset_and_step(frames, len(movie)) == (offset, step)


def test_movie_without_fluorescence_is_left_alone(example_frames):
    """A bright-field-only movie must not have a pattern invented for it."""
    bright, _ = example_frames
    movie = np.array(bright)

    assert fluorescence.detect_fluor_frames(movie) == []
    assert fluorescence.fluor_offset_and_step([], len(movie)) == (None, None)


def test_single_fluorescence_frame(example_frames):
    """One fluorescence frame has an offset but no meaningful step."""
    movie, wanted = build_movie(example_frames, 25, 12, 100)
    assert wanted == [12]

    frames = fluorescence.detect_fluor_frames(movie)

    assert frames == [12]
    assert fluorescence.fluor_offset_and_step(frames, len(movie)) == (12, None)


def test_fluorescence_brighter_than_brightfield(example_frames):
    """Detection keys on the two kinds of image differing, not on which is darker.

    This is the case that showed a 3x separation threshold was too strict: these
    frames separate by only about 2x.
    """
    movie, wanted = build_movie(example_frames, 30, 4, 6)
    movie = movie.copy()
    for frame_nb in wanted:
        movie[frame_nb] = np.clip(60000 - movie[frame_nb].astype(int),
                                  0, 65535).astype(np.uint16)

    assert fluorescence.detect_fluor_frames(movie) == wanted


def test_irregular_spacing_is_reported_exactly(example_frames):
    """Unevenly spaced frames are still found, even though no step describes them."""
    bright, fluor = example_frames
    wanted = [3, 9, 10, 20, 26]

    movie, b, f = [], 0, 0
    for i in range(30):
        if i in wanted:
            movie.append(fluor[f % len(fluor)]); f += 1
        else:
            movie.append(bright[b % len(bright)]); b += 1

    frames = fluorescence.detect_fluor_frames(np.array(movie))

    assert frames == wanted
    # a single offset/step cannot describe these, which is what main.py warns on
    offset, step = fluorescence.fluor_offset_and_step(frames, len(movie))
    assert frames != fluorescence.list_fluor_frames(len(movie), offset, step)


def test_too_short_to_judge():
    """Two frames are not enough to tell one kind of image from another."""
    assert fluorescence.detect_fluor_frames(np.zeros((2, 8, 8))) == []


# --------------------------------------------------------------------------
# the frame arithmetic itself
# --------------------------------------------------------------------------

def test_frame_list_never_runs_past_the_end():
    """Regression: the old hand-rolled loop walked one frame past the movie.

    On a 59 frame movie with -fo 5 -fs 6 it produced a final entry of 59, and
    the fluorescence step then died with
    `IndexError: index 59 is out of bounds for axis 0 with size 59`
    whenever the movie did not happen to end on a fluorescence frame.
    """
    frames = fluorescence.list_fluor_frames(59, 5, 6)

    assert max(frames) < 59
    assert frames == list(range(5, 59, 6))


@pytest.mark.parametrize('n_frames, offset, step', [
    (59, 5, 6), (364, 9, 6), (25, 9, 6), (100, 0, 3), (10, 9, 6),
])
def test_frame_list_stays_in_range(n_frames, offset, step):
    frames = fluorescence.list_fluor_frames(n_frames, offset, step)
    assert all(0 <= f < n_frames for f in frames)


def test_frame_list_rejects_a_nonsense_step():
    with pytest.raises(ValueError):
        fluorescence.list_fluor_frames(50, 0, 0)


def test_offset_and_step_round_trip():
    """What comes out of detection must describe the frames that went in."""
    expected = list(range(9, 364, 6))
    offset, step = fluorescence.fluor_offset_and_step(expected, 364)
    assert fluorescence.list_fluor_frames(364, offset, step) == expected


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
