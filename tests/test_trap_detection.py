"""Regression tests for finding the traps and building a template of one.

The template used to be cropped by hand, so a bad one was obvious to the person
who made it. Now that it is generated, nothing looks at it before the run
depends on it - hence these tests.

Run with `pytest tests` from the repository root.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trap_detection  # noqa: E402


def synthetic_array(v1, v2, shape=(600, 600), radius=9, noise=0.0, seed=0):
    """An image of identical blobs on the lattice spanned by v1 and v2."""
    rng = np.random.default_rng(seed)
    image = np.full(shape, 1000.0)

    grid = np.mgrid[:2 * radius + 1, :2 * radius + 1]
    blob = np.where(np.hypot(grid[0] - radius, grid[1] - radius) <= radius, 600.0, 0.0)

    for i in range(-20, 21):
        for j in range(-20, 21):
            x = int(shape[1] / 2 + i * v1[0] + j * v2[0])
            y = int(shape[0] / 2 + i * v1[1] + j * v2[1])
            if radius < x < shape[1] - radius - 1 and radius < y < shape[0] - radius - 1:
                image[y - radius:y + radius + 1, x - radius:x + radius + 1] += blob

    if noise:
        image += rng.normal(0, noise, shape)
    return image


def same_lattice(found, expected, tol=2):
    """A lattice vector is the same whichever way round it points."""
    for candidate in expected:
        for sign in (1, -1):
            if (abs(found[0] - sign * candidate[0]) <= tol
                    and abs(found[1] - sign * candidate[1]) <= tol):
                return True
    return False


@pytest.mark.parametrize('v1, v2', [
    ((80, 0), (0, 80)),          # square, no tilt
    ((109, 18), (5, 147)),       # the real device: tilted and oblique
    ((70, 25), (-25, 70)),       # rotated square
    ((90, 0), (0, 130)),         # rectangular
])
def test_lattice_is_recovered(v1, v2):
    image = synthetic_array(v1, v2)

    found = trap_detection.lattice_vectors(image)

    assert len(found) == 2
    for vector in found:
        assert same_lattice(vector, (v1, v2)), f'{vector} is neither {v1} nor {v2}'


def test_lattice_survives_noise():
    v1, v2 = (109, 18), (5, 147)
    image = synthetic_array(v1, v2, noise=120.0)

    for vector in trap_detection.lattice_vectors(image):
        assert same_lattice(vector, (v1, v2))


def test_template_is_centred_on_the_structure():
    """The blob must come out in the middle of the template, not at an edge."""
    v1, v2 = (109, 18), (5, 147)
    image = synthetic_array(v1, v2, radius=12)

    template, n_traps = trap_detection.build_template(image, v1, v2)

    assert n_traps > 4
    signal = template - np.median(template)
    total = signal.sum()
    rows, cols = np.mgrid[:template.shape[0], :template.shape[1]]
    centre_row = (signal * rows).sum() / total
    centre_col = (signal * cols).sum() / total

    assert abs(centre_row - template.shape[0] / 2) < 4
    assert abs(centre_col - template.shape[1] / 2) < 4


def test_template_removes_a_cell_sitting_in_one_trap():
    """A blemish on a single trap must not survive the median over all of them."""
    v1, v2 = (109, 18), (5, 147)
    clean = synthetic_array(v1, v2, radius=12)

    dirty = clean.copy()
    # a bright blob on one trap only, as a cell or a speck of dirt would be
    dirty[300 - 6:300 + 6, 300 - 6:300 + 6] += 5000.0

    from_clean, _ = trap_detection.build_template(clean, v1, v2)
    from_dirty, _ = trap_detection.build_template(dirty, v1, v2)

    assert np.abs(from_dirty - from_clean).max() < 0.05 * np.ptp(from_clean)


def test_background_drops_cells_the_masks_mark(tmp_path):
    """The cell-free background must show the device, not the cells on it."""
    import h5py
    import tifffile

    device = np.full((40, 40), 1000.0, dtype=np.float32)
    device[10:14, 10:14] = 4000.0                      # a fixed feature

    frames, masks = [], []
    for step in range(8):
        frame = device.copy()
        # a bright cell that moves, so no pixel is covered in every frame
        frame[20 + step, 20:24] = 9000.0
        mask = np.zeros_like(frame, dtype=np.int32)
        mask[20 + step, 20:24] = 1
        frames.append(frame); masks.append(mask)

    movie = tmp_path / 'movie.tif'
    tifffile.imwrite(movie, np.array(frames).astype(np.uint16))
    mask_file = tmp_path / 'mask.h5'
    with h5py.File(mask_file, 'w') as f:
        group = f.create_group('FOV0')
        for i, m in enumerate(masks):
            group.create_dataset('T' + str(i), data=m)

    background = trap_detection.cell_free_background(str(movie), str(mask_file))

    assert background[10:14, 10:14].min() > 3000      # device feature kept
    assert background[20:28, 20:24].max() < 2000      # cells gone


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
