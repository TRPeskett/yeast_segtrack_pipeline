"""Work out where the traps are, and what an empty one looks like, from the movie.

The template used to be cropped by hand from a frame in which the user had found
an empty, clean trap, and the inner dimensions of the trap measured by eye in
Preview. That is slow, has to be redone for every movie (the template does not
survive a change of tilt), and fails quietly: a template with a cell or a speck
of dirt in it matches badly, and the traps are then found badly.

The traps are a periodic array of identical structures, and by the time this runs
the whole movie has been segmented - so we know where the cells are. That is
enough to build a template with no cells and no dirt in it at all, without
anybody looking at anything.
"""

import os

import numpy as np
import h5py
import tifffile
from scipy import ndimage


def cell_free_background(movie_path, mask_path=None, n_samples=40):
    """A still image of the device with the cells taken out.

    For each pixel, the median over time of the frames in which no cell covered
    it. Cells move, divide and wash out, so almost every pixel is clear at some
    point; the few that never are (the middle of a trap holding one mother for
    the whole movie) fall back to the plain median over time, and the median over
    traps in build_template then removes those too.

    movie_path : the bright-field-only movie
    mask_path : segmentation of that movie. Without it this is just a median over
                time, which leaves long-lived cells behind.
    n_samples : how many frames to use. The median does not get meaningfully
                better with more, and every frame costs memory.
    """
    with tifffile.TiffFile(movie_path) as handle:
        n_frames = len(handle.pages)
        sample = np.linspace(0, n_frames - 1, min(n_samples, n_frames)).astype(int)
        stack = np.stack([handle.pages[i].asarray() for i in sample]).astype(np.float32)

    plain = np.median(stack, axis=0)

    if mask_path is None:
        return plain

    with h5py.File(mask_path, 'r') as f:
        group = list(f.keys())[0]
        cells = np.stack([f[group]['T' + str(i)][:] for i in sample]) > 0

    stack[cells] = np.nan
    with np.errstate(all='ignore'):
        background = np.nanmedian(stack, axis=0)

    return np.where(np.isfinite(background), background, plain)


def lattice_vectors(background, min_period=30):
    """The two shortest independent spacings of the trap array.

    Found from the autocorrelation, which peaks wherever the image maps onto
    itself. This picks up the tilt of the array for free, which is what makes a
    hand-made template non-transferable between movies.
    """
    signal = background - background.mean()
    signal = signal * (np.hanning(signal.shape[0])[:, None]
                       * np.hanning(signal.shape[1])[None, :])

    spectrum = np.fft.rfft2(signal)
    auto = np.fft.fftshift(np.fft.irfft2(spectrum * np.conj(spectrum), s=signal.shape))

    centre_y, centre_x = np.array(auto.shape) // 2
    rows, cols = np.ogrid[:auto.shape[0], :auto.shape[1]]
    radius = np.hypot(rows - centre_y, cols - centre_x)

    # the origin always peaks; we want the next two independent maxima
    work = np.where(radius < min_period, -np.inf, auto)

    vectors = []
    for _ in range(2):
        peak_y, peak_x = np.unravel_index(np.argmax(work), work.shape)
        vectors.append((int(peak_x - centre_x), int(peak_y - centre_y)))
        # blank this peak and the one opposite it, so the next is a new direction
        for x, y in ((peak_x, peak_y), (2 * centre_x - peak_x, 2 * centre_y - peak_y)):
            work[max(0, y - min_period):y + min_period,
                 max(0, x - min_period):x + min_period] = -np.inf

    return vectors


def _lattice_points(shape, origin, v1, v2, half):
    """Every lattice position sitting at least `half` px inside the image."""
    height, width = shape
    period = max(1.0, min(np.hypot(*v1), np.hypot(*v2)))
    span = int(2 * max(height, width) / period) + 2

    points = []
    for i in range(-span, span + 1):
        for j in range(-span, span + 1):
            x = origin[0] + i * v1[0] + j * v2[0]
            y = origin[1] + i * v1[1] + j * v2[1]
            if half <= x < width - half and half <= y < height - half:
                points.append((int(round(x)), int(round(y))))
    return points


def _median_tile(background, origin, v1, v2, size):
    half = size // 2
    tiles = [background[y - half:y - half + size, x - half:x - half + size]
             for x, y in _lattice_points(background.shape, origin, v1, v2, half + 1)]
    tiles = [t for t in tiles if t.shape == (size, size)]
    if not tiles:
        raise ValueError('the detected trap spacing does not fit inside the image')
    return np.median(np.stack(tiles), axis=0), len(tiles)


def _phase_from_cells(centroids, v1, v2, origin, bins=48):
    """Where the cells sit within one repeat of the lattice, as a pixel offset.

    Each cell centroid is written in units of the two lattice vectors and reduced
    modulo one repeat, so every trap's cells pile onto the same picture. The peak
    of that pile is where a cell sits relative to the array.
    """
    basis = np.array([[v1[0], v2[0]], [v1[1], v2[1]]], dtype=float)
    relative = np.linalg.solve(basis, (np.asarray(centroids) - np.asarray(origin)).T)
    fractional = np.mod(relative.T, 1.0)

    hist, _, _ = np.histogram2d(fractional[:, 0], fractional[:, 1],
                                bins=bins, range=[[0, 1], [0, 1]])
    # wrap the smoothing: the histogram lives on a torus
    hist = ndimage.gaussian_filter(hist, sigma=bins / 16, mode='wrap')
    peak_a, peak_b = np.unravel_index(np.argmax(hist), hist.shape)

    offset = basis @ np.array([(peak_a + 0.5) / bins, (peak_b + 0.5) / bins])
    return (origin[0] + offset[0], origin[1] + offset[1])


def build_template(background, v1, v2, size_factor=0.62):
    """A picture of an empty trap, as the median over every trap in the image.

    Any one trap may hold a cell or a speck of dirt; the median over all of them
    holds neither, because they are never in the same place twice.

    size_factor sets the template size as a fraction of the trap spacing. The
    trap crops that follow are twice the template, so this also sets how far each
    crop reaches into its neighbours. Detection is insensitive to it - every value
    from 0.5 to 0.78 finds the same traps at the same peak score of about 0.96 -
    so it is chosen small enough to keep the crops tight.

    Done in two passes. The first uses an arbitrary phase, which puts the trap
    somewhere in the tile but rarely in the middle; where it landed is then used
    to shift the lattice onto the traps, and the tiles are cut again. Rolling the
    first tile into place instead would be cheaper but wrong - the lattice is
    oblique, so opposite edges of a square tile do not meet.
    """
    period = min(np.hypot(*v1), np.hypot(*v2))
    origin = (background.shape[1] / 2, background.shape[0] / 2)

    coarse_size = int(round(period * 1.4))
    coarse, _ = _median_tile(background, origin, v1, v2, coarse_size)

    # Where in the tile does the structure sit? Blurring and taking the maximum
    # finds it. The blur has to be smooth rather than a flat disc: a disc wider
    # than the structure gives a plateau of equally good positions, and argmax
    # then returns whichever corner of that plateau it scans first, which put the
    # structure a quarter of a tile off centre.
    structure = np.abs(coarse - np.median(coarse))
    score = ndimage.gaussian_filter(structure, sigma=period * 0.15, mode='nearest')
    peak_y, peak_x = np.unravel_index(np.argmax(score), score.shape)

    # The blurred maximum is used as-is. Three "better" centrings were tried and
    # all were worse: the centre of mass of the structure (the two lobes are not
    # mirror images, so it sits 11 px off the waist), the peak of where cells sit
    # (the pocket extends to one side of the waist), and the bounding box of the
    # structure. Each of them moved the trap far enough off centre to cost real
    # mother traces. Measure before changing this.
    centred_origin = (origin[0] + peak_x - coarse_size // 2,
                      origin[1] + peak_y - coarse_size // 2)

    size = int(round(period * size_factor))
    template, n_traps = _median_tile(background, centred_origin, v1, v2, size)

    return template, n_traps


def cell_positions(mask_path, n_samples=20):
    """Centroids of every segmented cell, in (x, y), over a sample of frames."""
    if mask_path is None:
        return None

    from skimage.measure import regionprops

    with h5py.File(mask_path, 'r') as f:
        group = list(f.keys())[0]
        n_frames = len(f[group])
        sample = np.linspace(0, n_frames - 1, min(n_samples, n_frames)).astype(int)

        points = []
        for i in sample:
            for region in regionprops(f[group]['T' + str(i)][:].astype(int)):
                if region.area >= 50:
                    points.append((region.centroid[1], region.centroid[0]))

    return np.array(points) if points else None


def make_template(movie_path, mask_path, output_folder):
    """Build and save a template and the background it came from.

    Returns (template_path, background_path, background array).
    """
    background = cell_free_background(movie_path, mask_path)
    v1, v2 = lattice_vectors(background)

    period = min(np.hypot(*v1), np.hypot(*v2))
    print(f"  trap spacing {np.hypot(*v1):.0f} and {np.hypot(*v2):.0f} px "
          f"(array vectors {v1} and {v2})")

    template, n_traps = build_template(background, v1, v2)
    print(f"  template built from {n_traps} traps, {template.shape[1]}x{template.shape[0]} px")

    prefix = str(output_folder)
    os.makedirs(os.path.dirname(prefix) or '.', exist_ok=True)
    template_path = prefix + '_auto_template.tif'
    background_path = prefix + '_cell_free_background.tif'
    tifffile.imwrite(template_path, template.astype(np.uint16))
    tifffile.imwrite(background_path, background.astype(np.uint16))

    return template_path, background_path, background
