import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import glob
from PIL import Image, ImageSequence, ImageDraw, ImageFont
import math
import cv2
import os
import tifffile
from skimage.measure import regionprops
from skimage.segmentation import expand_labels


def measure_trap_centre(trap_paths, min_life=0.25, coverage=95.0, margin=1.15):
    """Size of the trap centre box, measured from the cell each trap is holding.

    The box decides which cells count as trapped. It used to be measured by hand
    off the template in Preview, which is fiddly and changes with the camera and
    objective. Instead, take the longest-lived cell in each trap - the cell the
    trap is holding, by definition - and measure how far its *pixels* reach from
    the middle of the trap.

    Pixels, not centroids. A held cell's centroid barely moves, so its scatter
    describes something far smaller than the pocket the cell sits in, and it is
    near enough isotropic - which produced a square box on a pocket that is
    twice as wide as it is tall, with a roof out in the trap pillar. The cell's
    own footprint has the shape of the space it is being held in.

    This is not circular: it runs after tracking, and the cell is picked purely on
    how long its track lasts, never on where it is. Note the discriminator really
    is track length and not how still the cell holds - a trapped mother's centroid
    drifts 10-30 px over a long movie as it grows, while a fragment of debris
    parked in the corner of the crop sits perfectly still.

    min_life : shortest track, as a fraction of the movie, that can count
    coverage : percentile of the held cell's pixel offsets the box must contain
    margin : slack on top, so a cell shifting a little is not suddenly an outsider
    :return: ((height, width), (offset_y, offset_x)) in px, both relative to the
             middle of the trap crop, or None if no trap held anything
    """
    half_extents, centres = [], []

    for path in trap_paths:
        table = path + '/midap/track_output_bayesian.csv'
        masks = path + '/midap/segmentations_bayesian.h5'
        if not (os.path.exists(table) and os.path.exists(masks)):
            continue

        with h5py.File(masks, 'r') as f:
            stack = f[list(f.keys())[0]][:]
        n_frames, height, width = stack.shape

        df = pd.read_csv(table)
        held = None
        for _track, group in df.groupby('trackID'):
            if len(group) >= min_life * n_frames and (held is None or len(group) > len(held)):
                held = group
        if held is None:
            continue

        rows, cols = [], []
        centre_rows, centre_cols, cell_half_rows, cell_half_cols = [], [], [], []
        for row in held.itertuples():
            frame_nb, label = int(row.frame), int(row.labelID)
            if frame_nb >= n_frames:
                continue
            yy, xx = np.where(stack[frame_nb] == label)
            if not len(yy):
                continue
            rows.append(yy - height / 2)
            cols.append(xx - width / 2)
            centre_rows.append(yy.mean() - height / 2)
            centre_cols.append(xx.mean() - width / 2)
            cell_half_rows.append((yy.max() - yy.min()) / 2)
            cell_half_cols.append((xx.max() - xx.min()) / 2)
        if not rows:
            continue

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)

        # How far the cell's centroid ranges, plus how big the cell itself is.
        # Taking the spread of the cell's pixels instead measures different
        # things on the two axes: the pocket squeezes the cell vertically, so the
        # pixels do describe its height, but nothing bounds the cell sideways, so
        # they badly under-describe its width. Splitting the two apart gives a
        # box that means the same thing along both axes.
        half_extents.append((
            np.percentile(np.abs(centre_rows - np.median(centre_rows)), coverage)
            + np.median(cell_half_rows),
            np.percentile(np.abs(centre_cols - np.median(centre_cols)), coverage)
            + np.median(cell_half_cols),
        ))
        centres.append((np.median(rows), np.median(cols)))

    if not half_extents:
        return None

    # Median across traps, so that a trap whose longest-lived cell is really
    # something lodged nearby cannot stretch the box on its own.
    half_extents = np.array(half_extents)
    half_height = float(np.median(half_extents[:, 0])) * margin
    half_width = float(np.median(half_extents[:, 1])) * margin

    # Where the box sits, not just how big it is. Pinning it to the middle of the
    # crop assumes the trap is exactly centred there, and a template centred a
    # few px out then puts every cell outside the box and finds no mothers at
    # all. Measuring the offset makes the box independent of how well the
    # template happened to be centred.
    offset = np.median(np.array(centres), axis=0)

    print(f"  trap centre measured from {len(half_extents)} traps, "
          f"offset {offset[1]:+.0f}, {offset[0]:+.0f} px from the middle of the crop")

    size = (max(4, int(round(2 * half_height))), max(4, int(round(2 * half_width))))
    return size, (float(offset[0]), float(offset[1]))


def save_trap_centre_check(template_path, trap_center_size, output_path,
                           trap_center_offset=(0.0, 0.0)):
    """Picture of the template with the trap centre box drawn on it, to eyeball."""
    template = np.asarray(tifffile.imread(template_path), dtype=float)
    scale = 6
    spread = np.ptp(template) or 1
    grey = ((template - template.min()) / spread * 255).astype(np.uint8)

    image = Image.fromarray(grey).convert('RGB').resize(
        (template.shape[1] * scale, template.shape[0] * scale), Image.NEAREST)

    height, width = template.shape
    box_h, box_w = trap_center_size
    left = ((width - box_w) / 2 + trap_center_offset[1]) * scale
    top = ((height - box_h) / 2 + trap_center_offset[0]) * scale

    draw = ImageDraw.Draw(image)
    draw.rectangle([left, top, left + box_w * scale, top + box_h * scale],
                   outline=(255, 60, 60), width=3)
    draw.text((6, 6), f'trap centre {box_w} x {box_h} px',
              fill=(255, 60, 60), font=load_font(18),
              stroke_width=1, stroke_fill=(0, 0, 0))

    image.save(output_path)
    return output_path


def _trapped_segments(tracking_table, max_gap):
    """Split each track's time inside the trap centre into unbroken segments.

    A segment is a run of frames in which one trackID sits in the middle of the
    trap, tolerating gaps of up to max_gap frames (a cell drifting briefly out of
    the centre box, or a frame the segmentation missed).
    """
    trapped = tracking_table[(tracking_table['Type'] == 'trapped')
                             & tracking_table['Centroid_x'].notna()]

    segments = []
    for track_id, group in trapped.groupby('trackID'):
        group = group.sort_values('frame')
        frames = group['frame'].astype(int).tolist()
        positions = list(zip(group['Centroid_y'], group['Centroid_x']))

        run_frames, run_positions = [frames[0]], [positions[0]]
        for frame_nb, position in zip(frames[1:], positions[1:]):
            if frame_nb - run_frames[-1] <= max_gap + 1:
                run_frames.append(frame_nb)
                run_positions.append(position)
            else:
                segments.append((int(track_id), run_frames, run_positions))
                run_frames, run_positions = [frame_nb], [position]
        segments.append((int(track_id), run_frames, run_positions))

    return sorted(segments, key=lambda seg: seg[1][0])


def follow_trap_occupant(tracking_table, trap_center_upper_left, trap_center_size,
                         max_jump=None, max_gap=5):
    """Follow the cell holding the middle of the trap through the movie.

    Returns {frame: trackID}.

    The cell is followed through segments rather than frame by frame. Picking the
    cell nearest the geometric centre of the trap each frame sounds equivalent,
    but it is not: a bud sitting momentarily closer to the middle than its mother
    hijacks the trace. Instead each track's unbroken time in the trap is one
    segment, and segments are joined only when the next one starts close in both
    time and space to where the last one ended - so a break in the tracker's own
    trackID does not end the mother trace, while a genuine handover to another
    cell, which moves the centroid a long way, does.

    Of all the ways the segments can be joined, the chain covering the most
    frames wins. With a single track trapped for the whole movie this reduces to
    the old behaviour: one segment, chosen because it is the longest.

    trap_center_upper_left : top-left corner of the trap centre box, (row, col)
    trap_center_size : (height, width) of the trap centre box, in px
    """
    if max_jump is None:
        max_jump = max(10.0, 0.5 * min(trap_center_size))

    segments = _trapped_segments(tracking_table, max_gap)
    if not segments:
        return {}

    # longest chain of joinable segments, walking backwards so each segment
    # already knows the best continuation available to it
    best_length = [0] * len(segments)
    best_next = [None] * len(segments)
    for i in range(len(segments) - 1, -1, -1):
        _, frames_i, positions_i = segments[i]
        best_length[i] = len(frames_i)
        for j in range(i + 1, len(segments)):
            _, frames_j, positions_j = segments[j]
            gap = frames_j[0] - frames_i[-1]
            if gap <= 0:
                continue          # overlaps in time, so a different cell
            if gap > max_gap + 1:
                break             # segments are sorted, nothing later is closer
            step = np.hypot(positions_j[0][0] - positions_i[-1][0],
                            positions_j[0][1] - positions_i[-1][1])
            if step <= max_jump and len(frames_i) + best_length[j] > best_length[i]:
                best_length[i] = len(frames_i) + best_length[j]
                best_next[i] = j

    chain = {}
    index = int(np.argmax(best_length))
    while index is not None:
        track_id, frames, _ = segments[index]
        for frame_nb in frames:
            chain[frame_nb] = track_id
        index = best_next[index]

    return chain


def id_the_mother(path: str, trap_center_size: list, trap_center_offset=(0.0, 0.0),
                  max_jump: float = None, max_gap: int = 5) -> None:
    '''
    Uses the csv file resulting from the tracking step to identify the mother cell.

    The mother is the cell that holds the middle of the trap. It is followed from
    frame to frame by position, so that it stays the mother across a break in its
    track: on a 364 frame movie the tracker rarely holds one cell for the whole
    run, and the previous rule - "the single trackID with the most frames in the
    middle of the trap" - then labelled only the longest surviving fragment. That
    left traps whose mother was present from frame 0 with a mother trace starting
    at frame 250 or later.

    path : path to the folder containing splitted traps
    trap_center_size : size, in px, of the center of the trap
    max_jump : how far, in px, the occupant of the trap may move between two
               frames and still be considered the same cell. Defaults to half the
               short side of the trap centre. A real jump means a different cell
               has taken the trap, and the chain stops there.
    max_gap : how many consecutive frames the middle of the trap may be empty (or
              the step rejected) before the mother is considered lost
    '''

    # first, read the necessary input datas.
    # The pandas package seems a convenient option for the csv file

    tracking_table = pd.read_csv(path + '/midap/track_output_bayesian.csv')
    tracking_table['Centroid_x'] = np.nan
    tracking_table['Centroid_y'] = np.nan

    with h5py.File(path + '/midap/segmentations_bayesian.h5', "r") as f:
        a_group_key = list(f.keys())
        dset = f[a_group_key[0]]
        masks = list(dset)

    image_size = np.shape(masks[0])
    offset_y, offset_x = trap_center_offset
    trap_center_upper_left = [(image_size[0] - trap_center_size[0]) / 2 + offset_y,
                              (image_size[1] - trap_center_size[1]) / 2 + offset_x]

    # in one loop, line by line from pandas :
    # compute centroids of each cell in each frame
    # see if cell inside the trap
    # re-save the dataframe

    # measure every cell of a frame in one pass instead of re-scanning the whole
    # frame once per row of the table
    centroid_lookup = {}
    for frame_nb, frame_data in enumerate(masks):
        for region in regionprops(np.asarray(frame_data).astype(int)):
            centroid_lookup[(frame_nb, region.label)] = region.centroid

    for index, row in tracking_table.iterrows():
        current_cell_id = int(row['labelID'])
        current_frame = int(row['frame'])

        centroids = centroid_lookup.get((current_frame, current_cell_id))
        if centroids is None:
            continue

        if (trap_center_upper_left[0] < centroids[0] < trap_center_upper_left[0] + trap_center_size[0] and
                trap_center_upper_left[1] < centroids[1] < trap_center_upper_left[1] + trap_center_size[1]):
            class_cell = 'trapped'
        else:
            class_cell = 'outsider'

        # put all of this nice data back in the df
        tracking_table.at[index, 'Centroid_x'] = centroids[1]
        tracking_table.at[index, 'Centroid_y'] = centroids[0]
        tracking_table.at[index, 'Type'] = class_cell

    # follow the cell holding the middle of the trap through time, across any
    # breaks in its track
    chain = follow_trap_occupant(tracking_table,
                                 trap_center_upper_left,
                                 trap_center_size,
                                 max_jump=max_jump,
                                 max_gap=max_gap)

    tracking_table['mother_trackID'] = np.nan
    for frame_nb, track_id in chain.items():
        rows = (tracking_table['frame'] == frame_nb) & (tracking_table['trackID'] == track_id)
        tracking_table.loc[rows, 'Type'] = 'mother'
        tracking_table.loc[rows, 'mother_trackID'] = track_id

    tracking_table.to_csv(path + '/tracking_with_mother.csv')


def generate_summary_plots(trap_path: str) -> None:
    '''
    Will generate two plats per trap : the evolution of the size of the
    mother cell with frame, and the evolution of the average fluorescence
    value with frame.

    trap_path : path to the directory containing the movie in splitted traps
    '''
    trap_nb = trap_path.split('/')[-1]

    full_df = pd.read_csv(trap_path + '/tracking_with_fluor.csv')
    df_inter = full_df.loc[(full_df['Type'] == 'mother') & (full_df['fluor_avg'] != -1.)]
    df_fluor_mother = df_inter[['frame', 'fluor_avg', 'fluor_std']].copy()
    df_fluor_val_mother = df_inter[['frame', 'fluorescence_values']].copy()

    fluor_stat_arr = df_fluor_mother.to_numpy()
    fluor_val_arr = df_fluor_val_mother.to_numpy()

    list_of_fluor_arrays = []
    for numframe, values in fluor_val_arr:
        reconstructed_list = []
        for val in values[1:-1].split(","):
            reconstructed_list.append(float(val))
        list_of_fluor_arrays.append(reconstructed_list)

    # the second value is a string, but we only need
    # the number of elements separated by the comma
    # = number of fluor px to get an indication on the cell size :

    size_cell = lambda x: [len(elmt[1].split(',')) for elmt in x]

    to_plot_size_cell = np.vstack((fluor_val_arr[:, 0],
                                   size_cell(fluor_val_arr)[:]))

    plt.rcParams["figure.figsize"] = (12, 10)
    fig, axs = plt.subplots(2)

    l1, = axs[0].plot(fluor_stat_arr[:, 0], fluor_stat_arr[:, 1])
    l2, = axs[0].plot(fluor_stat_arr[:, 0], fluor_stat_arr[:, 2])
    axs[0].set_title('Fluorescence values, trap nb. ' + trap_nb)
    axs[0].set_xlabel('Frame number')
    axs[0].set_ylabel('Average value or std, units from tif file')
    fig.legend((l1, l2), ('Average', 'Standard deviation'))

    axs[1].plot(to_plot_size_cell[0, :], to_plot_size_cell[1, :], color='tab:green')
    axs[1].set_title('Size of the cell')
    axs[1].set_xlabel('Frame number')
    axs[1].set_ylabel('Size of cell (number of px)')

    plt.savefig(trap_path + '/summary_trap_' + trap_nb + '.png')
    plt.close()


# Qualitative palette for the check movie, stable so a cell keeps its colour for
# its whole life. Cells within one trap get consecutive Global_ids and a trap can
# run to 60+ tracks over a long movie, so a 10 colour palette put several cells
# of the same trap on the same colour. Three qualitative maps are chained to give
# 60 slots, the most distinct ones first.
_PALETTE = (np.concatenate([plt.get_cmap(name).colors
                            for name in ('tab10', 'tab20b', 'tab20c', 'tab20')])
            * 255).astype(np.uint8)

# Cells that were segmented but belong to no tracked trap: shown, but dull, so
# they are clearly distinguishable from tracked cells.
_UNTRACKED_COLOUR = np.array([130, 130, 130], dtype=np.uint8)

MASK_ALPHA = 0.55


def _touching_label_pairs(mask, distance=3):
    """Pairs of labels whose regions come within `distance` px of each other.

    Cells in these images rarely share a border pixel, so the labels are grown
    slightly first; otherwise nearly nothing would count as adjacent and the
    recolouring below would never fire.
    """
    grown = expand_labels(mask, distance=distance)

    pairs = set()
    for a, b in ((grown[:, :-1], grown[:, 1:]), (grown[:-1, :], grown[1:, :])):
        differing = (a != b) & (a > 0) & (b > 0)
        for left, right in zip(a[differing], b[differing]):
            pairs.add((min(left, right), max(left, right)))

    return pairs


def assign_colours(mask, lookup, frame):
    """label -> RGB, colouring each cell by the identity it was tracked as.

    Colouring by Global_id rather than by position means a correctly tracked
    cell keeps one colour from the first frame to the last, so a tracking error
    shows up as a cell *changing colour* between frames - which is the thing
    that is hard to see when every cell is painted the same white.

    Where two touching cells would land on the same palette entry the *younger*
    one gives way. Seniority matters: when a bud appears next to its mother it is
    the bud that must move, otherwise a mother that has held one colour for
    hundreds of frames flips the moment it buds. The previous version broke ties
    on the segmentation label, which is renumbered every frame, so which of the
    two moved changed from frame to frame and both appeared to flicker.
    """
    labels = [int(r.label) for r in regionprops(np.asarray(mask).astype(int))]

    # preferred colour: fixed by identity, so it is stable across frames
    colour_ix = {}
    seniority = {}
    untracked = set()
    for label in labels:
        entry = lookup.get((frame, label))
        if entry is None:
            untracked.add(label)
        else:
            colour_ix[label] = int(entry['global_id']) % len(_PALETTE)
            seniority[label] = (int(entry['first_frame']), int(entry['global_id']))

    neighbours = {label: set() for label in colour_ix}
    for left, right in _touching_label_pairs(np.asarray(mask).astype(int)):
        if left in neighbours and right in neighbours:
            neighbours[left].add(right)
            neighbours[right].add(left)

    # oldest cell first, so it keeps its colour and newcomers work around it
    order = sorted(colour_ix, key=lambda label: seniority[label])
    settled = {}
    for label in order:
        taken = {settled[n] for n in neighbours[label] if n in settled}
        if colour_ix[label] in taken:
            free = [c for c in range(len(_PALETTE)) if c not in taken]
            settled[label] = free[0] if free else colour_ix[label]
        else:
            settled[label] = colour_ix[label]

    colours = {label: _PALETTE[ix] for label, ix in settled.items()}
    colours.update({label: _UNTRACKED_COLOUR for label in untracked})
    return colours


# Tried in order. ArialHB is Arial *Hebrew* - it happens to carry digits, which
# is all the old code needed, but it is a surprising default and not present off
# macOS, so try ordinary Latin faces first and degrade gracefully.
_FONT_CANDIDATES = (
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Geneva.ttf",
    "/System/Library/Fonts/ArialHB.ttc",
    "DejaVuSans.ttf",
)


def load_font(size):
    """First available truetype face at the requested size, else PIL's default."""
    for candidate in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def cell_label_font(mask, max_size=16, min_size=7):
    """Pick a font small enough that the id fits roughly inside a typical cell.

    The ids run into the hundreds while the cells are only ~20 px across, so a
    fixed 25 pt font drew the numbers of neighbouring cells straight over each
    other and hid the very thing you are trying to look at.
    """
    areas = [r.area for r in regionprops(np.asarray(mask).astype(int))]
    if areas:
        typical_diameter = 2 * np.sqrt(np.median(areas) / np.pi)
        size = int(np.clip(round(typical_diameter * 0.55), min_size, max_size))
    else:
        size = min_size

    return load_font(size)


def trap_boxes(output_path):
    """trap number -> (x_min, y_min, x_max, y_max) in full-frame coordinates.

    The traps are fixed features of the device, so their labels should be nailed
    to them. Deriving the anchor from the cells present in a frame instead makes
    the label jump around as cells move, divide and wash out.
    """
    boxes = {}
    for trap_dir in glob.glob(str(output_path) + '/split_data/*'):
        crops = [t for t in glob.glob(trap_dir + '/*.tif')
                 if 'shift_corrected' not in os.path.basename(t)]
        if not crops:
            continue
        parts = os.path.basename(crops[0]).split('_')
        try:
            boxes[int(os.path.basename(trap_dir))] = tuple(int(p) for p in parts[:4])
        except ValueError:
            continue
    return boxes


def render_check_frame(frame, mask, lookup, frame_nb, font=None, boxes=None):
    """One frame of the check movie: greyscale image under coloured cell masks."""
    # Scale by a percentile rather than the max. The fluorescence frames peak at
    # 65535, so dividing by the max rendered them at a mean brightness of about
    # 3/255 - effectively black - and a single hot pixel did the same to a
    # bright-field frame.
    low, high = np.percentile(frame, (1, 99))
    if high <= low:
        high = low + 1
    grey = np.clip((frame - low) / (high - low), 0, 1) * 255.
    canvas = np.repeat(grey[:, :, None], 3, axis=2).astype(np.float64)

    mask = np.asarray(mask)
    colours = assign_colours(mask, lookup, frame_nb)

    # blend the colour in only over the cells, leaving the background image at
    # full brightness instead of dimming the whole frame
    overlay = np.zeros_like(canvas)
    cells = mask > 0
    for label, colour in colours.items():
        overlay[mask == label] = colour
    canvas[cells] = canvas[cells] * (1 - MASK_ALPHA) + overlay[cells] * MASK_ALPHA

    image = Image.fromarray(canvas.astype(np.uint8), mode='RGB')
    draw = ImageDraw.Draw(image)

    # Only the cell number goes on the cell, so the label does not cover the cell
    # it belongs to. The trap number is drawn once, pinned to the trap's own crop
    # box, so it holds still for the whole movie.
    for region in regionprops(mask.astype(int)):
        entry = lookup.get((frame_nb, int(region.label)))
        if entry is None:
            continue
        row, col = region.centroid
        # white with a dark outline stays readable over both the colour patches
        # and the background
        draw.text((col, row), str(entry['cell_nb']), fill=(255, 255, 255), font=font,
                  anchor='mm', stroke_width=1, stroke_fill=(0, 0, 0))

    for trap, (x_min, y_min, x_max, y_max) in (boxes or {}).items():
        # centred over the trap rather than in the corner of its crop box: the
        # box carries a wide margin, so a corner anchor sits oddly far from the
        # trap it names
        draw.text(((x_min + x_max) / 2, y_min + 0.30 * (y_max - y_min)),
                  f'trap {trap}', fill=(255, 210, 60), font=font, anchor='mb',
                  stroke_width=1, stroke_fill=(0, 0, 0))

    return image


def check_segmentation_and_tracking(path_to_full_movie, path_to_first_mask, output_path):
    df = pd.read_csv(str(output_path) + '/all_traps_summary.csv')

    frames = tifffile.imread(path_to_full_movie)

    lookup = build_label_lookup(df)

    output_image = (str(output_path) + '/check_segmentation_and_tracking.tiff')

    # Write frame by frame rather than building the whole stack first. On a
    # 364-frame movie the rendered RGB frames alone are ~1.1 GB, on top of the
    # raw movie already in memory, and it would be a poor trade to lose a long
    # run at the very last step.
    font = None
    boxes = trap_boxes(output_path)
    with h5py.File(path_to_first_mask, "r") as f, \
            tifffile.TiffWriter(output_image) as writer:
        group = list(f.keys())[0]
        for i in range(0, len(f[group])):
            mask = f[group]['T' + str(i)][:]
            if font is None:
                font = cell_label_font(mask)
            rendered = render_check_frame(frames[i], mask, lookup, i, font=font,
                                          boxes=boxes)
            writer.write(np.asarray(rendered), photometric='rgb', contiguous=True)

    return 0


def build_label_lookup(df):
    """(frame, labelID) -> {global_id, trap_nb, name, first_frame}, built once.

    The previous code filtered the full summary table twice for every cell of
    every frame, which is why writing the check movie took longer than the
    tracking itself on movies with many traps.
    """
    first_frame = df.groupby('Global_id')['frame'].min().to_dict()

    lookup = {}
    for row in df.itertuples():
        lookup.setdefault((int(row.frame), int(row.labelID)), {
            'global_id': row.Global_id,
            'trap_nb': row.trap_nb,
            'cell_nb': getattr(row, 'Cell_nb', row.Global_id),
            'first_frame': first_frame.get(row.Global_id, row.frame),
        })
    return lookup


def summary_csv(path_to_splitted_traps):
    list_traps = glob.glob(str(path_to_splitted_traps) + '/split_data/*')

    all_df = []
    cpt_id = 0
    for trap in list_traps:
        trap_nb = int(trap.split('/')[-1])
        csv_path = trap + '/tracking_with_fluor.csv'
        if not os.path.exists(csv_path):
            # -no_fl skips the fluorescence step, and that step is what writes
            # tracking_with_fluor.csv - so on a bright-field-only movie every
            # trap looked like a tracking failure and the run died at the last
            # step with all of its segmentation and tracking already on disk.
            # The mother step writes its own csv holding everything except the
            # fluorescence columns, so fall back to that.
            fallback = trap + '/tracking_with_mother.csv'
            if not os.path.exists(fallback):
                print(f"  no tracking output for trap {trap_nb}, skipping it")
                continue
            csv_path = fallback
        one_df = pd.read_csv(csv_path)
        # the columns the fluorescence step would have added, carrying the
        # values it uses for a movie that has no fluorescence in it
        for column, when_absent in (('fluor_avg', -1.0), ('fluor_max', -1.0),
                                    ('fluor_std', -1.0),
                                    ('fluorescence_values', 'none')):
            if column not in one_df.columns:
                one_df[column] = when_absent
        one_df['trap_nb'] = trap_nb

        # Give every (trap, trackID) pair its own id. The previous version
        # carried the "last seen trackID" across traps and only assigned an id
        # when the trackID changed from one row to the next, so the first track
        # of the first trap never got one (it stayed 0) and any trap whose first
        # track happened to repeat the previous trap's last trackID was skipped
        # too, leaving several different cells sharing Global_id 0.
        one_df['Global_id'] = 0
        one_df['Cell_name'] = ''
        one_df['Cell_nb'] = 0
        for cell_nb, track_id in enumerate(one_df['trackID'].unique(), start=1):
            cpt_id += 1
            rows = one_df['trackID'] == track_id
            one_df.loc[rows, 'Global_id'] = cpt_id
            # short, readable, and still unique across the movie. Global_id runs
            # to several thousand on a long movie because it counts every track
            # in every trap, which makes for unwieldy labels on the check movie
            # and awkward numbers to quote when reporting a tracking error.
            # '-' not '.': '13.1' and '13.10' are the same number, so a dot
            # separator collapses distinct cells as soon as the csv is read back
            one_df.loc[rows, 'Cell_nb'] = cell_nb
            one_df.loc[rows, 'Cell_name'] = f'{trap_nb}-{cell_nb}'

        all_df.append(one_df)

    if not all_df:
        raise RuntimeError(
            "No trap produced any tracking output, so no summary can be built. "
            "Check the tracking step of the log for the reason.")

    df_res = pd.concat(all_df, ignore_index=True)
    df_res.to_csv(str(path_to_splitted_traps) + '/all_traps_summary.csv')
