"""
Draw the figures of the README that are not written by a training run.

  architecture   the layers of the model, with a real input, its latent vector, and what the model makes of it
  capture        clean renderings of training texts next to what the capture simulation turns them into
  size_sweep     identification accuracy against the size of the model
  identify       an image put through identify.py: the page that was found, its pieces, and the fonts found

Usage: python scripts/make_figures.py [--out docs/figures] [--photo path/to/photo.jpg] [--only NAME ...]

It needs what the model was trained with: the fonts and the configuration next to the exported model.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from font_reconstructor import factory, preprocessing  # noqa: E402
from font_reconstructor.inference import FontIdentifier  # noqa: E402
from font_reconstructor.logger import figures  # noqa: E402
from font_reconstructor.logger.figures import (  # noqa: E402
    AXIS, DIVERGING, GRID, INK, INK_MUTED, INK_SECONDARY, SERIES_1, SERIES_2, SURFACE, display_text)
from font_reconstructor.reporting import choose_panel, clean_input, glyph_masks  # noqa: E402
from font_reconstructor.utils import read_json  # noqa: E402

# results of the size sweep, read from the logs of the runs: 5 epochs of 2 million samples each. The largest
# run stopped after its fourth epoch.
SIZE_SWEEP = [
    # encoder channels, decoder width, encoder parameters, decoder parameters, top-1, top-5, epochs
    (4, 4, 20_580, 26_954, 0.4126, 0.7532, 5),
    (8, 9, 79_368, 80_034, 0.5289, 0.8625, 5),
    (12, 16, 176_396, 190_634, 0.5872, 0.9068, 5),
    (16, 22, 311_664, 319_130, 0.6081, 0.9225, 4),
]


def count(*modules):
    return sum(parameter.numel() for module in modules for parameter in module.parameters())


def box(ax, x, y, width, height, title, lines=(), color=AXIS, title_color=INK):
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle='round,pad=0,rounding_size=0.06', linewidth=1,
                                edgecolor=color, facecolor='#232322'))
    ax.text(x + width / 2, y + height - 0.17, title, color=title_color, fontsize=8.5, fontweight='bold', ha='center',
            va='top')
    for number, line in enumerate(lines):
        ax.text(x + width / 2, y + height - 0.42 - 0.2 * number, line, color=INK_SECONDARY, fontsize=7.5, ha='center',
                va='top')


def arrow(ax, start, end, color=INK_MUTED):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=9, linewidth=1, color=color,
                                 shrinkA=0, shrinkB=0))


def tiles(fingerprint, per_row=21):
    """
    the glyphs of a fingerprint side by side, as a grey image in [0, 1]
    """
    glyphs, height, width = fingerprint.shape
    rows = int(np.ceil(glyphs / per_row))
    sheet = np.full((rows * (height + 2) - 2, per_row * (width + 2) - 2), 0.1, dtype=np.float32)
    for glyph in range(glyphs):
        row, column = divmod(glyph, per_row)
        y, x = row * (height + 2), column * (width + 2)
        sheet[y:y + height, x:x + width] = np.clip((fingerprint[glyph] + 1.0) / 2.0, 0.0, 1.0)
    return sheet


def volume(figure, ax, lift, x, y, width, height, depth, face=None, cmap='gray', labels=(), label_y=None):
    """
    Draw a stack of feature maps as a block seen from the front left: its face is one map, its depth the
    number of maps.

    :param x, y: lower left corner of the face, in inches
    :param face: image for the face, values in [0, 1]
    :param labels: lines of text under the block, the first one in bold
    :param label_y: where the labels start. Right under the block by default.
    :return: (x of the right end of the block, y of the middle of its face)
    """
    dx, dy = 0.62 * depth, 0.36 * depth
    top = [(x, y + height), (x + width, y + height), (x + width + dx, y + height + dy), (x + dx, y + height + dy)]
    side = [(x + width, y), (x + width + dx, y + dy), (x + width + dx, y + height + dy), (x + width, y + height)]
    ax.add_patch(Polygon(top, closed=True, facecolor='#3a3a37', edgecolor=AXIS, linewidth=0.8))
    ax.add_patch(Polygon(side, closed=True, facecolor='#262625', edgecolor=AXIS, linewidth=0.8))
    ax.add_patch(Polygon([(x, y), (x + width, y), (x + width, y + height), (x, y + height)], closed=True,
                         facecolor='#2f2f2d', edgecolor=AXIS, linewidth=0.8))
    if face is not None:
        inset = figure.add_axes(figures._rect(figure, x, y + lift, width, height))
        inset.imshow(face, cmap=cmap, vmin=0.0, vmax=1.0, aspect='auto', interpolation='nearest')
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_color(AXIS)
    label_y = y - 0.16 if label_y is None else label_y
    for number, line in enumerate(labels):
        ax.text(x + (width + dx) / 2, label_y - 0.19 * number, line, color=INK if number == 0 else INK_SECONDARY,
                fontsize=8 if number == 0 else 7.5, fontweight='bold' if number == 0 else 'normal', ha='center',
                va='top')
    return x + width + dx, y + height / 2


def activation(tensor):
    """
    the mean of a stack of feature maps over its channels, scaled to [0, 1], as the face of its block
    """
    image = tensor[0].abs().mean(dim=0).cpu().numpy()
    return (image - image.min()) / max(float(image.max() - image.min()), 1e-9)


@torch.no_grad()
def architecture(identifier, config, out, sample_number=0):
    model, device = identifier.model, identifier.device
    fonts = factory.build_fontset(config)
    corpus = factory.build_corpus(config)
    _, valid_loader = factory.build_train_valid_loaders(config, fonts, device, corpus, num_workers=0)
    dataset = valid_loader.dataset
    # the first sample of the validation panel whose font the model gets right
    for position in choose_panel(dataset, 8)[sample_number:]:
        sample = dataset[position]
        image = sample['image'].unsqueeze(0).to(device)
        latent = model.encode(image)
        matches = identifier.index.search(latent.cpu().numpy(), k=3)
        if matches[0]['name'] == sample['font']:
            break
    output = model.decode(latent)[0].cpu().numpy()
    target = sample['target'].numpy()
    style = F.softmax(model.predict_style(latent), dim=1)[0].cpu()
    seen, _ = glyph_masks(fonts, [sample['text']], [sample['font_index']], target.shape[0])
    lift = 1.3  # room below the layers for the two fingerprints
    size = (16.4, 9.3)

    figure = figures._figure(
        size[0], size[1] + lift, 'Model',
        'A text image is encoded to a latent vector of 32 numbers, which identifies the font. Every block is a '
        'stack of feature maps: its face shows their mean for this image, its depth is their number.')
    ax = figure.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, size[0])
    ax.set_ylim(-lift, size[1])
    ax.set_axis_off()

    def picture(image, x, y, width, height, cmap='gray', vmin=0.0, vmax=1.0):
        inset = figure.add_axes(figures._rect(figure, x, y + lift, width, height))
        inset.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', interpolation='nearest')
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_color(AXIS)
        return inset

    def face_size(height, width, unit):
        # sizes shrink more slowly than the feature maps do, so that the small ones stay visible
        return unit * (width / 64) ** 0.62, unit * (height / 64) ** 0.62

    def depth(channels):
        return 0.105 * channels ** 0.5

    pitch = 1.55  # least distance between the blocks of a row

    # encoder: the feature maps after the first convolution and after every residual block
    row = 5.5
    ax.text(0.5, 8.05, 'INPUT', color=INK_MUTED, fontsize=8, fontweight='bold')
    ax.text(0.5, 7.8, f"a text set in {sample['font']}, a font held out of training, after the simulated capture",
            color=INK_SECONDARY, fontsize=8)
    ax.text(0.5, 7.42, f"ENCODER   {count(model.encoder, model.to_latent):,} parameters", color=INK_MUTED, fontsize=8,
            fontweight='bold')
    width, height = face_size(64, 256, 0.62)
    labels_at = row - 0.42
    end, middle = volume(figure, ax, lift, 0.5, row, width, height, 0.03, (sample['image'][0].numpy() + 1) / 2,
                         labels=['input', '1 \u00d7 64 \u00d7 256', display_text(sample['text'])], label_y=labels_at)
    tensor = image
    stages = [('conv 3\u00d73', model.encoder[:3])] + [(f'residual block {n}', model.encoder[2 + n:3 + n]) for n in range(1, 5)]
    for title, layers in stages:
        tensor = layers(tensor)
        channels, rows, columns = tensor.shape[1:]
        width, height = face_size(rows, columns, 0.62)
        # every block gets room for its label, however small it is
        x = end + 0.3 + max(0.0, pitch - 0.3 - width - 0.62 * depth(channels)) / 2
        arrow(ax, (end + 0.04, middle), (x - 0.04, middle))
        block_end, _ = volume(figure, ax, lift, x, middle - height / 2, width, height, depth(channels),
                              activation(tensor), cmap=figures.MAGNITUDE, label_y=labels_at,
                              labels=[title, f'{channels} \u00d7 {rows} \u00d7 {columns}', f'{count(layers):,} parameters'])
        end = max(block_end, end + pitch)
    # average pooling and the projection to the latent vector
    latent_x, latent_width = 12.9, 2.9
    arrow(ax, (end + 0.04, middle), (latent_x - 0.06, middle), color=SERIES_1)
    ax.text((end + latent_x) / 2, middle + 0.12, 'average over the image,\nlinear', color=INK_SECONDARY, fontsize=7.5,
            ha='center', va='bottom')
    ax.text((end + latent_x) / 2, middle - 0.12, f'{count(model.to_latent):,} parameters', color=INK_SECONDARY,
            fontsize=7.5, ha='center', va='top')

    # latent vector
    values = latent[0].cpu().numpy()
    ax.text(latent_x, middle + 0.42, f'LATENT VECTOR   {model.latent_dim} numbers', color=SERIES_1, fontsize=8,
            fontweight='bold')
    picture(values[None, :], latent_x, middle - 0.17, latent_width, 0.34, cmap=DIVERGING, vmin=-np.abs(values).max(),
            vmax=np.abs(values).max())
    ax.text(latent_x, middle - 0.27, 'its values for this image', color=INK_MUTED, fontsize=7.5, va='top')

    # font index
    ax.text(latent_x, 8.05, 'FONT INDEX', color=INK_MUTED, fontsize=8, fontweight='bold')
    ax.text(latent_x, 7.8, f'cosine similarity to each of {len(identifier.index):,} fonts', color=INK_SECONDARY, fontsize=8)
    for number, match in enumerate(matches):
        right = match['name'] == sample['font']
        ax.text(latent_x, 7.4 - 0.26 * number, f"{match['rank']}.  {match['name']}", color=INK if right else INK_SECONDARY,
                fontsize=8.5, fontweight='bold' if right else 'normal')
        ax.text(latent_x + latent_width, 7.4 - 0.26 * number, f"{match['similarity']:.2f}", color=INK_SECONDARY,
                fontsize=8.5, ha='right')
    arrow(ax, (latent_x + latent_width / 2, middle + 0.62), (latent_x + latent_width / 2, 6.72), color=SERIES_1)

    # style head
    best = int(style.argmax())
    prediction = f'{identifier.info["style_names"][best]} ({float(style[best]):.0%})'
    box(ax, latent_x + 0.75, 3.75, latent_width - 0.75, 0.95, 'style head',
        [f'linear, {count(model.style_head):,} parameters', prediction])
    arrow(ax, (latent_x + 1.8, middle - 0.5), (latent_x + 1.8, 4.72), color=SERIES_1)

    # decoder, from the latent vector back to the left
    row = 2.35
    ax.text(0.5, 4.05, f"DECODER   {count(model.from_latent, model.decoder):,} parameters, used in training only",
            color=INK_MUTED, fontsize=8, fontweight='bold')
    tensor = model.from_latent(latent)
    decoder_stages = [('linear', model.from_latent, tensor)]
    for number in range(4):
        tensor = model.decoder[number](tensor)
        decoder_stages.append((f'resize + conv {number + 1}', model.decoder[number], tensor))
    tensor = model.decoder[4:](tensor)
    decoder_stages.append(('conv 3\u00d73, tanh', model.decoder[4], tensor))

    start = latent_x + 0.05
    ax.plot([latent_x + 0.3, latent_x + 0.3], [middle - 0.5, row + 0.3], color=SERIES_1, linewidth=1)
    for number, (title, module, tensor) in enumerate(decoder_stages):
        channels, rows, columns = tensor.shape[1:]
        width, height = face_size(rows, columns, 1.0)
        block_width = width + 0.62 * depth(channels)
        slot = max(block_width + 0.3, pitch)
        x = start - 0.3 - (slot - 0.3 - block_width) / 2 - block_width
        last = number == len(decoder_stages) - 1
        # the last block is the fingerprint itself: its face shows one of the glyphs it holds
        face = (output[int(np.argmax(seen[0].numpy()))] + 1) / 2 if last else activation(tensor)
        volume(figure, ax, lift, x, row + 0.3 - height / 2, width, height, depth(channels), face,
               cmap='gray' if last else figures.MAGNITUDE, label_y=row - 0.42,
               labels=[title, f'{channels} \u00d7 {rows} \u00d7 {columns}', f'{count(module):,} parameters'])
        arrow(ax, (latent_x + 0.3 if number == 0 else start - 0.04, row + 0.3), (x + block_width + 0.04, row + 0.3),
              color=SERIES_1 if number == 0 else INK_MUTED)
        start -= slot
        output_x = x

    # the fingerprint the decoder draws and the one it should draw
    sheet_width = 11.2
    sheet_height = sheet_width * (2 * 66 - 2) / (21 * 66 - 2)
    sheet_x = 1.9
    picture(tiles(output), sheet_x, 0.1, sheet_width, sheet_height)
    picture(tiles(target), sheet_x, 0.1 - sheet_height - 0.12, sheet_width, sheet_height)
    ax.plot([output_x - 0.3, output_x - 0.55, output_x - 0.55], [row + 0.3, row + 0.3, 0.1 + sheet_height + 0.25],
            color=INK_MUTED, linewidth=1)
    arrow(ax, (output_x - 0.55, 0.1 + sheet_height + 0.3), (output_x - 0.55, 0.1 + sheet_height + 0.03))
    ax.text(0.5, 0.1 + sheet_height / 2, 'OUTPUT\nthe fingerprint\nthe model draws', color=INK_MUTED, fontsize=7.5,
            va='center')
    ax.text(0.5, 0.1 - sheet_height / 2 - 0.12, 'TARGET\nthe glyphs of\nthe font', color=INK_MUTED, fontsize=7.5,
            va='center')
    shown = int(seen[0].sum())
    ax.text(sheet_x + sheet_width + 0.2, 0.1 + sheet_height / 2,
            f'42 glyphs of 64 \u00d7 64 pixels.\n{shown} of them occur\nin the input text.', color=INK_SECONDARY, fontsize=7.5,
            va='center')
    return figures.save_figure(figure, out, dpi=160)


def capture(identifier, config, out, pairs=12):
    fonts = factory.build_fontset(config)
    corpus = factory.build_corpus(config)
    _, valid_loader = factory.build_train_valid_loaders(config, fonts, identifier.device, corpus, num_workers=0)
    dataset = valid_loader.dataset
    dims = identifier.dims
    figure = figures._figure(12.6, 0.95 + 0.78 * (pairs // 2), 'Training images',
                             'Left of each pair: the clean rendering of a text. Right: the same text after the '
                             'simulated capture, which is what the model is given.')
    positions = np.random.default_rng(5).choice(len(dataset), pairs, replace=False)
    for number, position in enumerate(positions):
        row, column = divmod(number, 2)
        y = 0.25 + 0.78 * (pairs // 2 - 1 - row)
        images = (clean_input(dataset, int(position), dims), dataset[int(position)]['image'][0].numpy())
        for part, image in enumerate(images):
            inset = figure.add_axes(figures._rect(figure, 0.5 + column * 6.0 + part * 2.9, y, 2.8, 0.7))
            inset.imshow((image + 1) / 2, cmap='gray', vmin=0, vmax=1, aspect='auto')
            inset.set_axis_off()
    return figures.save_figure(figure, out, dpi=160)


def size_sweep(out):
    figure = figures._figure(7.6, 4.6, 'Accuracy against model size',
                             'Held-out fonts, each identified among 1,493. Encoder and decoder are about the same size.')
    ax = figures._axes(figure, figures._rect(figure, 0.8, 0.75, 5.6, 2.85))
    sizes = np.array([row[2] for row in SIZE_SWEEP])
    for column, (label, color) in enumerate((('top-1', SERIES_1), ('top-5', SERIES_2)), start=4):
        values = np.array([row[column] for row in SIZE_SWEEP])
        ax.plot(sizes, values, color=color, linewidth=2, marker='o', markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=2, solid_capstyle='round')
        ax.annotate(f"{label}  {values[-1]:.1%}", (sizes[-1], values[-1]), xytext=(10, 0), textcoords='offset points',
                    color=INK, fontsize=9, va='center', annotation_clip=False)
        for size, value in zip(sizes[:-1], values[:-1]):
            ax.annotate(f"{value:.1%}", (size, value), xytext=(0, 9), textcoords='offset points', color=INK_SECONDARY,
                        fontsize=8, ha='center')
    ax.set_xscale('log')
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{size / 1000:.0f} k" for size in sizes])
    ax.minorticks_off()
    ax.set_xlim(sizes[0] * 0.8, sizes[-1] * 1.25)
    ax.set_ylim(0.3, 1.0)
    ax.set_yticks(np.linspace(0.4, 1.0, 4))
    ax.set_yticklabels([f"{value:.0%}" for value in np.linspace(0.4, 1.0, 4)])
    ax.set_xlabel('parameters of the encoder, on a logarithmic scale')
    figure.text(0.8 / 7.6, 0.2 / 4.6, 'Five epochs of two million samples each. The largest model stopped after four.',
                color=INK_MUTED, fontsize=8)
    ax.grid(axis='y', color=GRID, linewidth=1)
    return figures.save_figure(figure, out, dpi=160)


def identify(identifier, photo, out, truth=None):
    """
    :param truth: optional names of what each line shows, for the table
    """
    pixels = preprocessing.load_grey(photo)
    page, corners = preprocessing.find_page(pixels)
    ink = preprocessing.clean_page(photo)
    result = identifier.identify(photo, k=3)

    height = 6.2
    photo_width = height * pixels.shape[1] / pixels.shape[0]
    page_width = height * ink.shape[1] / ink.shape[0]
    figure = figures._figure(0.5 + photo_width + 0.4 + page_width + 0.4 + 6.4, height + 1.45, 'From a photo to fonts',
                             'identify.py finds the page, straightens it, cleans it and cuts the lines into pieces. '
                             'Every line is then looked up in the font index.')
    ax = figure.add_axes(figures._rect(figure, 0.5, 0.45, photo_width, height))
    ax.imshow(pixels, cmap='gray', vmin=0, vmax=255)
    if corners is not None:
        ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=SERIES_1, linewidth=2))
    ax.set_axis_off()
    ax.set_title('photo, with the page that was found', loc='left', fontsize=9, color=INK_SECONDARY, pad=6)

    ax = figure.add_axes(figures._rect(figure, 0.5 + photo_width + 0.4, 0.45, page_width, height))
    ax.imshow(ink, cmap='gray', vmin=0, vmax=1)
    for piece in result.pieces:
        left, top, right, bottom = piece.box
        ax.add_patch(Polygon([(left, top), (right, top), (right, bottom), (left, bottom)], closed=True, fill=False,
                             edgecolor=SERIES_1, linewidth=1.2))
    ax.set_axis_off()
    ax.set_title(f'straightened and cleaned, {len(result.pieces)} pieces', loc='left', fontsize=9, color=INK_SECONDARY,
                 pad=6)

    left = 0.5 + photo_width + 0.4 + page_width + 0.4
    ax = figure.add_axes(figures._rect(figure, left, 0.45, 6.4, height))
    ax.set_xlim(0, 6.4)
    ax.set_ylim(ink.shape[0], 0)
    ax.set_axis_off()
    ax.set_title('best matches of each line, with their cosine similarity', loc='left', fontsize=9,
                 color=INK_SECONDARY, pad=6)
    tops = {}
    for piece in result.pieces:
        tops.setdefault(piece.line, []).append((piece.box[1] + piece.box[3]) / 2)
    for line in result.lines:
        y = float(np.mean(tops[line['line']]))
        if truth and line['line'] < len(truth):
            ax.text(0.0, y - 0.022 * ink.shape[0], f"the line says: {truth[line['line']]}", color=INK_MUTED,
                    fontsize=7.5, va='center')
        for number, match in enumerate(line['matches'][:3]):
            row = y + 0.012 * ink.shape[0]
            ax.text(number * 2.15, row, f"{match['name'][:22]}", color=INK if number == 0 else INK_SECONDARY,
                    fontsize=8.5, fontweight='bold' if number == 0 else 'normal', va='center')
            ax.text(number * 2.15 + 1.95, row, f"{match['similarity']:.2f}", color=INK_MUTED, fontsize=8, va='center',
                    ha='right')
    return figures.save_figure(figure, out, dpi=140)


def main():
    parser = argparse.ArgumentParser(description='Draw the figures of the README')
    parser.add_argument('--model', default='pretrained/font_encoder.pth')
    parser.add_argument('--index', default='pretrained/fonts.sqlite')
    parser.add_argument('--config', default='pretrained/training_config.json')
    parser.add_argument('--out', default='docs/figures')
    parser.add_argument('--photo', help='image for the identify figure')
    parser.add_argument('--truth', nargs='*', help='what each line of the photo shows, for the identify figure')
    parser.add_argument('--only', nargs='*', default=['architecture', 'capture', 'size_sweep', 'identify'])
    args = parser.parse_args()

    out = Path(args.out)
    identifier = FontIdentifier(args.model, args.index)
    config = read_json(args.config)
    if 'architecture' in args.only:
        print(architecture(identifier, config, out / 'architecture.png'))
    if 'capture' in args.only:
        print(capture(identifier, config, out / 'capture.png'))
    if 'size_sweep' in args.only:
        print(size_sweep(out / 'size_sweep.png'))
    if 'identify' in args.only and args.photo:
        print(identify(identifier, Image.open(args.photo), out / 'identify.png', args.truth))


if __name__ == '__main__':
    main()
