"""Fine-tune the YeaZ U-Net on hand-corrected trap crops.

The shipped weights do not transfer to the 2023 imaging condition, so the model
is adapted rather than replaced: training starts from the existing bright-field
weights at a low learning rate. With a few dozen crops that is the only sensible
option - training a U-Net of this size from scratch on that much data would not
work.

The network ends in a sigmoid and predicts one channel, cell against background,
which `yeaz.unet.segment.segment` then splits into individual cells by watershed.
So the target is a binary mask and the loss is plain binary cross entropy; the
corrected masks are binarised, and whatever numbering they carry is ignored.

Validation is split by microscope position, never at random. Crops from the same
position and nearby frames show the same cells in the same trap, so a random
split would put near-duplicates on both sides and report a score that has
nothing to do with how the model behaves on a movie it has not seen.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main  # noqa: E402


def load_corrected(stems):
    """(images, masks, index) for every crop marked corrected across some rounds."""
    images, masks, rows = [], [], []
    for stem in stems:
        index = pd.read_csv(stem + '_index.csv')
        if 'corrected' not in index.columns:
            continue
        image_stack = tifffile.imread(stem + '_images.tif')
        label_stack = tifffile.imread(stem + '_labels.tif')
        for i, row in index.iterrows():
            if not row['corrected']:
                continue
            images.append(image_stack[i])
            masks.append((label_stack[i] > 0).astype(np.float32))
            rows.append({**row.to_dict(), 'round': os.path.basename(stem)})
    if not images:
        raise SystemExit(
            'no corrected crops found. Run retrain/annotate.py and mark crops '
            'with "d" before training.')
    return np.array(images), np.array(masks), pd.DataFrame(rows)


def split_by_position(index, holdout_fraction=0.25, seed=0):
    """Train/validation indices, keeping whole positions on one side or the other."""
    groups = index.groupby(['experiment', 'position']).indices
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    n_holdout = max(1, int(round(len(keys) * holdout_fraction)))
    validation_keys = set(keys[:n_holdout])

    train, validation = [], []
    for key, rows in groups.items():
        (validation if key in validation_keys else train).extend(rows)
    return np.array(sorted(train)), np.array(sorted(validation)), validation_keys


def augment(image, mask, rng):
    """Flips and right-angle turns, plus a little brightness and contrast.

    Only transforms that a microscope could plausibly have produced: the traps
    have no preferred orientation, but stretching or shearing would teach the
    model shapes that do not occur.
    """
    k = rng.integers(4)
    if k:
        image, mask = np.rot90(image, k), np.rot90(mask, k)
    if rng.random() < 0.5:
        image, mask = image[::-1], mask[::-1]
    if rng.random() < 0.5:
        image, mask = image[:, ::-1], mask[:, ::-1]

    image = image.astype(np.float32)
    image = image * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05) * image.std()
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def prepare(image):
    """Exactly the preprocessing inference uses, so training matches deployment."""
    return main.normalise_frame(image.astype(np.uint16)).astype(np.float32)


def border_weight(shape, border=8):
    """1 everywhere except a border band, which is 0 and so contributes no loss.

    A crop cuts through cells at its edges. Those cells are whole in the movie -
    the crop boundary is an artefact of how the training data was made, not
    something the model will meet at inference, where it sees the full frame.

    Painting a half-visible cell is correct, since those pixels really are cell.
    But whether a two-pixel sliver at the edge is a cell or a speck is often
    genuinely undecidable, and a person should not have to rule on it. Excluding
    the band means whatever is or is not painted out there teaches nothing, so
    the annotation rule can simply be "paint what you can see and do not worry
    about the edge".
    """
    weight = np.zeros(shape, dtype=np.float32)
    if border > 0:
        weight[border:-border, border:-border] = 1.0
    else:
        weight[:] = 1.0
    return weight


def masked_bce(prediction, target, weight, eps=1e-7):
    """Binary cross entropy averaged over the weighted pixels only.

    `weight` covers one crop, but a whole batch arrives at once, so the sum has
    to be divided by the number of images as well. Without that the loss - and
    the gradient with it - grows in proportion to the batch, which ran training
    at four times the requested learning rate and made the validation loss (one
    batch of 18) look five times worse than the training loss (batches of 4)
    when the two were in fact comparable.
    """
    prediction = prediction.clamp(eps, 1 - eps)
    loss = -(target * prediction.log() + (1 - target) * (1 - prediction).log())
    n_images = prediction.shape[0] if prediction.dim() > 2 else 1
    return (loss * weight).sum() / (weight.sum().clamp(min=1.0) * n_images)


def iou(prediction, truth, threshold=0.9, border=8):
    if border > 0:
        prediction = prediction[border:-border, border:-border]
        truth = truth[border:-border, border:-border]
    predicted = prediction > threshold
    actual = truth > 0.5
    union = np.logical_or(predicted, actual).sum()
    return float(np.logical_and(predicted, actual).sum() / union) if union else 1.0


def finetune(stems, output, epochs=40, learning_rate=1e-5, batch_size=4,
             holdout_fraction=0.25, seed=0, border=8):
    images, masks, index = load_corrected(stems)
    train_ix, val_ix, val_keys = split_by_position(index, holdout_fraction, seed)

    print(f'{len(images)} corrected crops from {index.groupby(["experiment","position"]).ngroups} positions')
    print(f'  training on {len(train_ix)}, validating on {len(val_ix)}')
    print(f'  held-out positions: {sorted(val_keys)}')
    print(f'  cell pixels: {100 * masks.mean():.1f}% of all pixels')

    device = main.pick_device('cuda')
    from yeaz.unet.model_pytorch import UNet
    model = UNet()
    model.load_state_dict(torch.load(
        os.getenv('WEIGHTS_YEAST', 'weights/weights_budding_BF_multilab_0_1'),
        map_location='cpu'))
    model = model.to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    weight = torch.from_numpy(border_weight(images.shape[1:], border)).to(device)
    print(f'  ignoring a {border} px border in the loss '
          f'({100 * (1 - float(weight.mean())):.0f}% of each crop)')

    prepared = np.stack([prepare(im) for im in images])

    validation_x = torch.from_numpy(prepared[val_ix][:, None]).float()
    validation_y = torch.from_numpy(masks[val_ix][:, None]).float()

    best = None
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train_ix)
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            xs, ys = [], []
            for i in batch:
                x, y = augment(prepared[i], masks[i], rng)
                xs.append(x); ys.append(y)
            x = torch.from_numpy(np.stack(xs)[:, None]).float().to(device)
            y = torch.from_numpy(np.stack(ys)[:, None]).float().to(device)

            optimiser.zero_grad()
            loss = masked_bce(model(x), y, weight)
            loss.backward()
            optimiser.step()
            total += float(loss) * len(batch)

        model.eval()
        with torch.no_grad():
            predicted = model(validation_x.to(device)).cpu().numpy()[:, 0]
        val_loss = float(masked_bce(torch.from_numpy(predicted[:, None]),
                                    validation_y, weight.cpu()))
        scores = [iou(predicted[i], masks[val_ix][i], border=border)
                  for i in range(len(val_ix))]
        mean_iou = float(np.mean(scores))

        marker = ''
        if best is None or mean_iou > best[0]:
            best = (mean_iou, epoch)
            torch.save(model.state_dict(), output)
            marker = '  <- saved'
        print(f'  epoch {epoch:3d}  train {total / max(len(order),1):.4f}  '
              f'val {val_loss:.4f}  IoU {mean_iou:.3f}{marker}')

    print(f'\nbest validation IoU {best[0]:.3f} at epoch {best[1]}  ->  {output}')
    return best


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rounds', nargs='*', default=None,
                        help='round stems; defaults to every round found')
    parser.add_argument('--output', default='weights/weights_2023_finetuned')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--border', type=int, default=8,
                        help='px of crop border excluded from the loss')
    args = parser.parse_args()

    stems = args.rounds or sorted(
        p[:-len('_index.csv')] for p in glob.glob('retrain/rounds/*_index.csv'))
    finetune(stems, args.output, epochs=args.epochs, learning_rate=args.lr,
             border=args.border)
