"""
The figures a run writes to tensorboard. They share one look: a dark surface, text in neutral ink, one blue hue
for magnitudes, blue and orange where two series are compared, and green and red only for right and wrong.

Every function takes plain arrays and returns a matplotlib figure. Nothing here knows about models or loaders.
"""
from typing import List, Optional, Sequence

import arabic_reshaper
import matplotlib
import numpy as np
from bidi.algorithm import get_display
from matplotlib import ft2font
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

# surfaces and ink
SURFACE = '#1a1a19'
INK = '#ffffff'
INK_SECONDARY = '#c3c2b7'
INK_MUTED = '#898781'
GRID = '#2c2c2a'
AXIS = '#383835'
# series colors, validated together for color vision deficiencies on the dark surface
SERIES_1 = '#3987e5'
SERIES_2 = '#d95926'
# status colors, only used together with an icon and a label
GOOD = '#0ca30c'
CRITICAL = '#d03b3b'
# one-hue ramps for magnitudes, the low end recedes into the surface
MAGNITUDE = LinearSegmentedColormap.from_list('magnitude', [SURFACE, '#184f95', '#3987e5', '#cde2fb'])
ERROR = LinearSegmentedColormap.from_list('error', ['#000000', '#7a2f12', '#d95926', '#fbd9c9'])
# two hues around a neutral grey for values with a sign, the same number of steps on each side
DIVERGING = LinearSegmentedColormap.from_list(
    'diverging', ['#cde2fb', '#3987e5', '#1c5cab', '#383835', '#a83a3a', '#e66767', '#f8cfcf'])

_GAP = 2  # pixels of surface between the glyphs of a sheet
_STYLE = {
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'text.color': INK,
    'axes.facecolor': SURFACE,
    'axes.edgecolor': AXIS,
    'axes.labelcolor': INK_SECONDARY,
    'axes.titlecolor': INK,
    'xtick.color': INK_MUTED,
    'ytick.color': INK_MUTED,
    'xtick.labelcolor': INK_SECONDARY,
    'ytick.labelcolor': INK_SECONDARY,
    'figure.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'axes.grid': False,
}


# matplotlib shapes text itself since it is built with the Raqm layout library: it joins the letters of Arabic
# script and writes them from right to left. Older versions draw every character as it comes, from left to right.
MATPLOTLIB_SHAPES_TEXT = hasattr(ft2font, '__libraqm_version__')


def display_text(text: str) -> str:
    """
    Prepare text for matplotlib so that Arabic and Persian words come out joined and in reading order.

    A matplotlib that shapes text itself gets the text unchanged. Doing the work for it would be undone: it
    would reverse the already reversed text and break the joins. Older versions get the letters replaced by
    their joined forms and put in visual order.
    """
    if MATPLOTLIB_SHAPES_TEXT:
        return str(text)
    return get_display(arabic_reshaper.reshape(str(text)))


def _figure(width: float, height: float, title: str, subtitle: str = '') -> Figure:
    """
    a figure with the title block all figures share. It is not registered with pyplot, so it needs no closing.
    """
    with matplotlib.rc_context(_STYLE):
        figure = Figure(figsize=(width, height), dpi=100)
    figure.set_facecolor(SURFACE)
    figure.text(0.5 / width, 1 - 0.32 / height, title, color=INK, fontsize=13, fontweight='bold', va='center')
    if subtitle:
        figure.text(0.5 / width, 1 - 0.62 / height, subtitle, color=INK_SECONDARY, fontsize=9, va='center')
    return figure


def _axes(figure: Figure, rect, grid_axis: Optional[str] = 'y'):
    """
    axes in the shared look: recessive hairline grid, no box, text in secondary ink
    """
    with matplotlib.rc_context(_STYLE):
        ax = figure.add_axes(rect)
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)
    baseline = 'bottom' if grid_axis != 'x' else 'left'
    ax.spines[baseline].set_visible(True)
    ax.spines[baseline].set_color(AXIS)
    ax.tick_params(length=0, colors=INK_SECONDARY, labelsize=9)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=1, linestyle='-')
        ax.set_axisbelow(True)
    return ax


def _rect(figure: Figure, left: float, bottom: float, width: float, height: float):
    """
    a rectangle given in inches as a fraction of the figure
    """
    total_width, total_height = figure.get_size_inches()
    return [left / total_width, bottom / total_height, width / total_width, height / total_height]


def _image_axes(figure: Figure, rect):
    ax = figure.add_axes(rect)
    ax.set_axis_off()
    return ax


def _grey(image: np.ndarray) -> np.ndarray:
    """
    an image in [-1, 1] as an rgb array, always on the same scale so that weak outputs look weak
    """
    level = np.clip((np.asarray(image, dtype=np.float32) + 1.0) / 2.0, 0.0, 1.0)
    return np.repeat(level[..., None], 3, axis=-1)


def _rgb(color: str) -> np.ndarray:
    return np.array(matplotlib.colors.to_rgb(color), dtype=np.float32)


def glyph_sheet(strips: Sequence[np.ndarray], kinds: Sequence[str], seen: Optional[np.ndarray] = None,
                per_row: int = 21) -> np.ndarray:
    """
    Lay out fingerprints as a sheet of glyph tiles: every strip is a block of rows with `per_row` glyphs each.

    :param strips: arrays (glyphs, height, width). A 'glyphs' strip holds images in [-1, 1], an 'error' strip
        the absolute difference of two such images.
    :param kinds: 'glyphs' or 'error' for each strip
    :param seen: bool array (glyphs,). A blue bar is drawn under the glyphs of the first strip it marks.
    :return: rgb array of the sheet
    """
    glyphs, height, width = strips[0].shape
    rows = int(np.ceil(glyphs / per_row))
    marker = 3 if seen is not None else 0
    row_height = height + _GAP
    strip_height = rows * row_height + 4
    first_strip_height = rows * (row_height + marker + (_GAP if marker else 0)) + 4
    total_height = first_strip_height + (len(strips) - 1) * strip_height
    sheet = np.tile(_rgb(SURFACE), (total_height, per_row * (width + _GAP) - _GAP, 1))

    top = 0
    for index, (strip, kind) in enumerate(zip(strips, kinds)):
        step = row_height + (marker + _GAP if index == 0 and marker else 0)
        for glyph in range(glyphs):
            row, column = divmod(glyph, per_row)
            y, x = top + row * step, column * (width + _GAP)
            if kind == 'error':
                tile = ERROR(np.clip(np.asarray(strip[glyph], dtype=np.float32) / 2.0, 0.0, 1.0))[..., :3]
            else:
                tile = _grey(strip[glyph])
            sheet[y:y + height, x:x + width] = tile
            if index == 0 and marker and seen[glyph]:
                sheet[y + height + _GAP:y + height + _GAP + marker, x:x + width] = _rgb(SERIES_1)
        top += first_strip_height if index == 0 else strip_height
    return sheet


def _strip_label_positions(strips: int, glyphs: int, height: int, seen: bool, per_row: int = 21) -> List[float]:
    """
    vertical centre of each strip of a glyph sheet, as a fraction from the top
    """
    rows = int(np.ceil(glyphs / per_row))
    marker = 3 + _GAP if seen else 0
    first = rows * (height + _GAP + marker) + 4
    other = rows * (height + _GAP) + 4
    total = first + (strips - 1) * other
    centres, top = [], 0
    for index in range(strips):
        size = first if index == 0 else other
        centres.append((top + size / 2) / total)
        top += size
    return centres


def _status(ax, x: float, y: float, correct: bool, label: str, fontsize: float = 9):
    """
    an icon in the status color followed by its label in ink, so that the color never stands alone
    """
    ax.text(x, y, '✓' if correct else '✗', color=GOOD if correct else CRITICAL, fontsize=fontsize + 2,
            fontweight='bold', va='center', ha='left', transform=ax.transAxes)
    ax.text(x + 0.07, y, label, color=INK, fontsize=fontsize, va='center', ha='left', transform=ax.transAxes)


def _sheet_size(strips: int, glyphs: int, height: int, width: int, seen: bool, per_row: int = 21):
    """
    (width, height) in inches of a glyph sheet shown at one screen pixel per sheet pixel
    """
    rows = int(np.ceil(glyphs / per_row))
    marker = 3 + _GAP if seen else 0
    pixels_high = rows * (height + _GAP + marker) + 4 + (strips - 1) * (rows * (height + _GAP) + 4)
    pixels_wide = per_row * (width + _GAP) - _GAP
    return pixels_wide / 100, pixels_high / 100


def _sample_rows(figure: Figure, rows: int, row_height: float, top_margin: float = 0.95):
    """
    the bottom edge in inches of each row of a figure of sample rows
    """
    total = figure.get_size_inches()[1]
    return [total - top_margin - (index + 1) * row_height for index in range(rows)]


def plot_panel(samples: Sequence[dict], num_fonts: Optional[int] = None) -> Figure:
    """
    The fixed panel of validation samples: what the model saw, what it identified and what it drew.

    Each sample is a dict with
        font, text          name of the font and the rendered text
        clean, input        arrays (height, width) in [-1, 1]: the ideal rendering and the image given to the model
        target, output      arrays (glyphs, height, width) in [-1, 1]
        seen                bool array (glyphs,), the glyphs that occur in the text
        error               reconstruction error of the sample
        skill               optional, share of the baseline's error the model removes
        rank                optional, rank of the true font among all fonts
        matches             optional, list of (font name, similarity) of the best matching fonts
        style, style_prediction, style_confidence   optional, true and predicted style
    """
    text_width, input_width = 3.5, 2.6
    sheet_width, sheet_height = _sheet_size(3, *np.shape(samples[0]['target']), seen=True)
    row_height = sheet_height + 0.35
    figure = _figure(
        0.5 + text_width + input_width + 0.6 + sheet_width + 0.4, 1.1 + row_height * len(samples),
        'Validation panel', 'The same held out fonts every epoch. A blue bar marks the glyphs that occur in the '
        'input text. The error row is brighter where output and target differ more.')

    for sample, bottom in zip(samples, _sample_rows(figure, len(samples), row_height)):
        height = row_height - 0.35

        # what was identified
        ax = _image_axes(figure, _rect(figure, 0.5, bottom, text_width - 0.3, height))
        ax.text(0, 0.94, display_text(sample['font']), color=INK, fontsize=11, fontweight='bold', va='center',
                transform=ax.transAxes)
        ax.text(0, 0.80, 'text', color=INK_MUTED, fontsize=8, va='center', transform=ax.transAxes)
        ax.text(0.13, 0.80, display_text(sample['text']), color=INK_SECONDARY, fontsize=10, va='center',
                transform=ax.transAxes)
        y = 0.64
        if sample.get('rank') is not None:
            of = f" of {num_fonts:,}" if num_fonts else ''
            _status(ax, 0, y, sample['rank'] == 1, f"rank {sample['rank']:,}{of}")
            y -= 0.115
            for position, (name, similarity) in enumerate(sample.get('matches', []), start=1):
                is_true = name == sample['font']
                ax.text(0.07, y, f"{position}  {display_text(name)[:26]}", fontsize=8, va='center',
                        color=INK if is_true else INK_SECONDARY, fontweight='bold' if is_true else 'normal',
                        transform=ax.transAxes)
                ax.text(1.0, y, f"{similarity:.2f}", color=INK_MUTED, fontsize=8, va='center', ha='right',
                        transform=ax.transAxes)
                y -= 0.095
            y -= 0.03
        if sample.get('style_prediction') is not None:
            label = f"style {sample['style']}"
            if sample['style_prediction'] != sample['style']:
                label += f", predicted {sample['style_prediction']}"
            label += f" ({sample['style_confidence']:.0%})"
            _status(ax, 0, y, sample['style_prediction'] == sample['style'], label)
            y -= 0.125
        reconstruction = f"reconstruction error {sample['error']:.3f}"
        if sample.get('skill') is not None:
            reconstruction += f", skill {sample['skill']:+.2f}"
        ax.text(0, y, reconstruction, color=INK_SECONDARY, fontsize=9, va='center', transform=ax.transAxes)

        # what the model saw
        left = 0.5 + text_width
        for label, image, offset in (('clean rendering', sample['clean'], height * 0.52),
                                     ('as given to the model', sample['input'], 0.0)):
            ax = _image_axes(figure, _rect(figure, left, bottom + offset, input_width - 0.3, height * 0.42))
            ax.imshow(_grey(image), aspect='equal', interpolation='nearest')
            ax.set_title(label, color=INK_MUTED, fontsize=8, loc='left', pad=3)

        # what it drew
        error = np.abs(np.asarray(sample['output']) - np.asarray(sample['target']))
        sheet = glyph_sheet([sample['target'], sample['output'], error], ['glyphs', 'glyphs', 'error'],
                            seen=sample['seen'])
        left += input_width + 0.6
        ax = _image_axes(figure, _rect(figure, left, bottom, sheet_width, height))
        ax.imshow(sheet, aspect='equal', interpolation='nearest')
        centres = _strip_label_positions(3, sample['target'].shape[0], sample['target'].shape[1], seen=True)
        for name, centre in zip(('target', 'output', 'error'), centres):
            ax.text(-0.012, 1 - centre, name, color=INK_MUTED, fontsize=8, va='center', ha='right',
                    transform=ax.transAxes)
    return figure


def plot_worst_cases(cases: Sequence[dict], num_fonts: Optional[int] = None) -> Figure:
    """
    The validation samples whose true font ranked worst, next to the font they were taken for.

    Each case is a dict with font, text, input (height, width), target and predicted (glyphs, height, width)
    in [-1, 1], predicted_font, rank and similarity. If the two fingerprints look alike, the confusion is one
    between near-duplicate fonts.
    """
    text_width, input_width = 3.5, 2.6
    shape = np.shape(cases[0]['target']) if len(cases) else (42, 32, 32)
    sheet_width, sheet_height = _sheet_size(3, *shape, seen=False)
    row_height = sheet_height + 0.35
    figure = _figure(
        0.5 + text_width + input_width + 0.6 + sheet_width + 0.4, 1.1 + row_height * max(len(cases), 1),
        'Worst identified samples', 'The samples whose true font ranked lowest this epoch, with the fingerprint '
        'of the font they were taken for. The difference row is brighter where the two fonts differ more.')

    for case, bottom in zip(cases, _sample_rows(figure, len(cases), row_height)):
        height = row_height - 0.35
        ax = _image_axes(figure, _rect(figure, 0.5, bottom, text_width - 0.3, height))
        ax.text(0, 0.90, display_text(case['font']), color=INK, fontsize=11, fontweight='bold', va='center',
                transform=ax.transAxes)
        ax.text(0, 0.72, 'text', color=INK_MUTED, fontsize=8, va='center', transform=ax.transAxes)
        ax.text(0.13, 0.72, display_text(case['text']), color=INK_SECONDARY, fontsize=10, va='center',
                transform=ax.transAxes)
        of = f" of {num_fonts:,}" if num_fonts else ''
        _status(ax, 0, 0.50, case['rank'] == 1, f"rank {case['rank']:,}{of}")
        ax.text(0, 0.30, 'taken for', color=INK_MUTED, fontsize=8, va='center', transform=ax.transAxes)
        ax.text(0, 0.14, f"{display_text(case['predicted_font'])[:30]}  ({case['similarity']:.2f})",
                color=INK_SECONDARY, fontsize=9, va='center', transform=ax.transAxes)

        left = 0.5 + text_width
        ax = _image_axes(figure, _rect(figure, left, bottom + height * 0.28, input_width - 0.3, height * 0.55))
        ax.imshow(_grey(case['input']), aspect='equal', interpolation='nearest')
        ax.set_title('as given to the model', color=INK_MUTED, fontsize=8, loc='left', pad=3)

        difference = np.abs(np.asarray(case['predicted']) - np.asarray(case['target']))
        sheet = glyph_sheet([case['target'], case['predicted'], difference], ['glyphs', 'glyphs', 'error'])
        left += input_width + 0.6
        ax = _image_axes(figure, _rect(figure, left, bottom, sheet_width, height))
        ax.imshow(sheet, aspect='equal', interpolation='nearest')
        centres = _strip_label_positions(3, case['target'].shape[0], case['target'].shape[1], seen=False)
        for name, centre in zip(('true font', 'taken for', 'difference'), centres):
            ax.text(-0.012, 1 - centre, name, color=INK_MUTED, fontsize=8, va='center', ha='right',
                    transform=ax.transAxes)
    return figure


def _mean(values: np.ndarray) -> float:
    """
    the mean of the values that are numbers, NaN if there are none
    """
    values = np.asarray(values, dtype=np.float64)
    return float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else float('nan')


def plot_glyph_error(glyphs: Sequence[str], seen: np.ndarray, unseen: np.ndarray) -> Figure:
    """
    Reconstruction error of each glyph, when it occurs in the input text and when the model has to infer it.

    :param glyphs: the character of each glyph
    :param seen, unseen: arrays (glyphs,) of mean errors, NaN where there is no sample
    """
    seen, unseen = np.asarray(seen, dtype=np.float64), np.asarray(unseen, dtype=np.float64)
    figure = _figure(
        max(8.0, 0.32 * len(glyphs) + 2.4), 4.9, 'Reconstruction error per glyph',
        f"Mean absolute error on held out fonts. In the input text {_mean(seen):.3f}, "
        f"not in the input text {_mean(unseen):.3f}.")
    ax = _axes(figure, _rect(figure, 0.8, 0.75, figure.get_size_inches()[0] - 1.3, 2.75))

    x = np.arange(len(glyphs))
    ax.vlines(x, np.fmin(seen, unseen), np.fmax(seen, unseen), color=AXIS, linewidth=2, zorder=2)
    marker = dict(marker='o', linestyle='none', markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    ax.plot(x, unseen, color=SERIES_2, label='not in the input text', **marker)
    ax.plot(x, seen, color=SERIES_1, label='in the input text', **marker)

    ax.set_xticks(x)
    ax.set_xticklabels(glyphs, fontsize=11)
    ax.set_xlim(-0.8, len(glyphs) - 0.2)
    highest = np.nanmax(np.concatenate([seen, unseen])) if np.isfinite(np.concatenate([seen, unseen])).any() else 1.0
    ax.set_ylim(0, highest * 1.15)
    ax.set_ylabel('mean absolute error')
    legend = ax.legend(loc='lower left', bbox_to_anchor=(-0.01, 1.03), ncol=2, frameon=False, fontsize=9,
                       handletextpad=0.2, columnspacing=1.6)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    return figure


def plot_accuracy_by_rank(ranks: np.ndarray, accuracy: np.ndarray, marks: Sequence[int] = (1, 5, 10),
                          num_fonts: Optional[int] = None) -> Figure:
    """
    Share of the validation samples whose true font is among the k best matches, for every k.
    """
    of = f" among {num_fonts:,} fonts" if num_fonts else ''
    figure = _figure(7.2, 4.4, 'Font identification by rank',
                     f"Share of held out samples whose true font is within the k best matches{of}.")
    ax = _axes(figure, _rect(figure, 0.8, 0.75, 5.9, 2.65))
    ax.plot(ranks, accuracy, color=SERIES_1, linewidth=2, solid_capstyle='round', solid_joinstyle='round')

    for k in marks:
        if k <= len(ranks):
            value = accuracy[k - 1]
            ax.plot([k], [value], marker='o', markersize=8, color=SERIES_1, markeredgecolor=SURFACE,
                    markeredgewidth=2, zorder=3)
            ax.annotate(f"top-{k}  {value:.1%}", (k, value), xytext=(8, -12), textcoords='offset points',
                        color=INK, fontsize=9)

    ax.set_xscale('log')
    ticks = [k for k in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) if k <= len(ranks)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(k) for k in ticks])
    ax.minorticks_off()
    ax.set_xlim(0.9, len(ranks) * 1.05)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_yticklabels([f"{value:.0%}" for value in np.linspace(0, 1, 5)])
    ax.set_xlabel('k, on a logarithmic scale')
    return figure


def plot_bars(names: Sequence[str], values: np.ndarray, counts: Optional[np.ndarray], title: str,
              subtitle: str, count_label: str = 'samples') -> Figure:
    """
    One horizontal bar per group, all in the same hue, with the value at the end of each bar.

    :param values: shares in [0, 1]
    :param counts: number of samples behind each bar, shown beside the name
    """
    figure = _figure(7.2, 1.55 + 0.42 * len(names), title, subtitle)
    ax = _axes(figure, _rect(figure, 2.3, 0.55, 4.2, 0.42 * len(names)), grid_axis='x')

    y = np.arange(len(names))[::-1]
    ax.barh(y, values, height=0.5, color=SERIES_1, linewidth=0)
    for position, value in zip(y, values):
        ax.text(value + 0.015, position, f"{value:.1%}", color=INK, fontsize=9, va='center')

    labels = [str(name) for name in names]
    if counts is not None:
        labels = [f"{name}   {count:,} {count_label}" for name, count in zip(labels, counts)]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.12)
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.set_xticklabels([f"{value:.0%}" for value in np.linspace(0, 1, 5)])
    ax.set_ylim(-0.6, len(names) - 0.4)
    return figure


def plot_confusion(names: Sequence[str], matrix: np.ndarray) -> Figure:
    """
    How the style head answers for each true style: each row shows the shares of the predicted styles.

    :param matrix: array (styles, styles) of sample counts, rows are true styles, columns predicted styles
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    # styles that neither occur nor are ever predicted would only add empty rows and columns
    used = (matrix.sum(axis=1) > 0) | (matrix.sum(axis=0) > 0)
    matrix, names = matrix[used][:, used], [name for name, keep in zip(names, used) if keep]
    totals = matrix.sum(axis=1, keepdims=True)
    shares = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)

    size = 0.72 * len(names)
    figure = _figure(size + 3.4, size + 2.5, 'Style predictions',
                     'Rows are the true style, columns the predicted style, as shares of each row.')
    ax = _axes(figure, _rect(figure, 2.6, 0.5, size, size), grid_axis=None)
    ax.spines['bottom'].set_visible(False)
    ax.imshow(shares, cmap=MAGNITUDE, vmin=0, vmax=1, aspect='equal')

    # a gap of surface color separates the cells
    ax.set_xticks(np.arange(len(names) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(names) + 1) - 0.5, minor=True)
    ax.grid(which='minor', color=SURFACE, linewidth=2)
    ax.tick_params(which='minor', length=0)

    for row in range(len(names)):
        for column in range(len(names)):
            share = shares[row, column]
            if share >= 0.005:
                ax.text(column, row, f"{share:.0%}", ha='center', va='center', fontsize=9,
                        color='#0b0b0b' if share > 0.75 else INK)

    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=35, ha='left')
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels([f"{name}   {int(total):,}" for name, total in zip(names, totals[:, 0])])
    return figure


def plot_latent_health(spread: np.ndarray, norms: np.ndarray) -> Figure:
    """
    Whether the latent space is used: the spread of each dimension over the validation samples, and the
    lengths of the latent vectors.

    A dimension with almost no spread carries no information. Lengths that drift far apart make the
    reconstruction depend on something the cosine similarity ignores.
    """
    spread, norms = np.asarray(spread), np.asarray(norms)
    idle = int((spread < 0.05 * np.median(spread)).sum())
    figure = _figure(
        11.0, 4.7, 'Latent space',
        f"{len(spread)} dimensions, {idle} of them nearly unused. "
        f"Vector length {norms.mean():.2f} on average, from {norms.min():.2f} to {norms.max():.2f}.")

    ax = _axes(figure, _rect(figure, 0.8, 0.75, 5.3, 2.6))
    ax.bar(np.arange(len(spread)), spread, width=0.6, color=SERIES_1, linewidth=0)
    ax.set_xlim(-0.8, len(spread) - 0.2)
    ax.set_xlabel('latent dimension')
    ax.set_ylabel('standard deviation over the samples')
    ax.set_title('Spread of each dimension', loc='left', fontsize=10, color=INK_SECONDARY, pad=8)

    ax = _axes(figure, _rect(figure, 7.0, 0.75, 3.5, 2.6))
    ax.hist(norms, bins=30, color=SERIES_1, rwidth=0.82, linewidth=0)
    ax.set_xlabel('length of the latent vector')
    ax.set_ylabel('samples')
    ax.set_title('Lengths of the latent vectors', loc='left', fontsize=10, color=INK_SECONDARY, pad=8)
    return figure


def _scale_legend(figure: Figure, rect, cmap, low: float, high: float, labels: Sequence[str]):
    """
    a short horizontal color scale with a label at each end and one in the middle
    """
    ax = figure.add_axes(rect)
    ax.imshow(np.linspace(low, high, 128)[None, :], cmap=cmap, vmin=low, vmax=high, aspect='auto')
    ax.set_yticks([])
    ax.set_xticks([0, 63.5, 127])
    tick_labels = ax.set_xticklabels(labels, fontsize=8, color=INK_SECONDARY)
    # the end labels stay inside the scale instead of hanging over its ends
    tick_labels[0].set_horizontalalignment('left')
    tick_labels[-1].set_horizontalalignment('right')
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_latent_comparison(latent: np.ndarray, groups: Sequence[int], fonts: Sequence[str], texts: Sequence[str],
                           mean: np.ndarray, spread: np.ndarray, font_signal: np.ndarray) -> Figure:
    """
    Latent vectors side by side, feature by feature, with the similarity of every pair of them.

    The rows are a few fonts with several texts each. Left: every latent dimension of every row, relative to
    the average of the validation set and in units of its spread. A column that keeps its color within a font
    and changes between fonts tells fonts apart. A column that changes within a font reacts to the text. The
    bars above say the same over the whole validation set. Right: the cosine similarity of every pair of rows.

    :param latent: array (rows, latent_dim)
    :param groups: the font group of each row, rows of one group are adjacent
    :param fonts, texts: font name and text of each row
    :param mean, spread: mean and standard deviation of each latent dimension over the validation set
    :param font_signal: share of the variation of each dimension that lies between fonts, in [0, 1]
    """
    latent = np.asarray(latent, dtype=np.float64)
    rows, dims = latent.shape
    groups = np.asarray(groups)
    relative = (latent - mean) / np.maximum(spread, 1e-9)

    unit = latent / np.maximum(np.linalg.norm(latent, axis=1, keepdims=True), 1e-12)
    similarity = unit @ unit.T
    same_font = groups[:, None] == groups[None, :]
    off_diagonal = ~np.eye(rows, dtype=bool)
    within = similarity[same_font & off_diagonal]
    between = similarity[~same_font]
    summary = ''
    if len(within) and len(between):
        summary = (f" Cosine similarity {within.mean():.2f} between texts of one font, "
                   f"{between.mean():.2f} between fonts.")

    cell_width, cell_height, label_width = 0.27, 0.25, 3.3
    map_width, map_height = dims * cell_width, rows * cell_height
    matrix_size = rows * 0.21
    figure = _figure(
        0.5 + label_width + map_width + 1.1 + matrix_size + 0.5, 1.2 + 1.35 + map_height + 1.0,
        'Latent vectors compared',
        'Each row is one image, grouped by font. A column that keeps its color within a font and changes '
        'between fonts tells fonts apart.' + summary)
    bottom, left = 0.95, 0.5 + label_width
    boundaries = [index - 0.5 for index in range(1, rows) if groups[index] != groups[index - 1]]

    # how much each feature tells fonts apart, over the whole validation set
    ax = _axes(figure, _rect(figure, left, bottom + map_height + 0.3, map_width, 0.9))
    ax.bar(np.arange(dims), font_signal, width=0.6, color=SERIES_1, linewidth=0)
    ax.set_xlim(-0.5, dims - 0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['0%', '50%', '100%'], fontsize=8)
    ax.set_xticks([])
    ax.set_title('Share of each feature\'s variation that lies between fonts, over the validation set',
                 loc='left', fontsize=9, color=INK_SECONDARY, pad=6)

    # every feature of every row
    ax = _axes(figure, _rect(figure, left, bottom, map_width, map_height), grid_axis=None)
    ax.spines['bottom'].set_visible(False)
    ax.imshow(relative, cmap=DIVERGING, vmin=-3, vmax=3, aspect='auto', interpolation='nearest')
    ax.set_xticks(np.arange(dims + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(rows + 1) - 0.5, minor=True)
    ax.grid(which='minor', color=SURFACE, linewidth=2)
    ax.tick_params(which='minor', length=0)
    for boundary in boundaries:
        ax.axhline(boundary, color=SURFACE, linewidth=7)
    ax.set_xticks(np.arange(0, dims, 4))
    ax.set_xticklabels([str(index) for index in range(0, dims, 4)], fontsize=8)
    ax.set_xlabel('latent feature')
    ax.set_yticks([])
    previous = None
    for row in range(rows):
        if groups[row] != previous:
            ax.text(-0.6 - label_width / cell_width + 0.2, row, f"{groups[row] + 1}  {display_text(fonts[row])[:24]}",
                    color=INK, fontsize=9, fontweight='bold', va='center', ha='left')
            previous = groups[row]
        ax.text(-0.8, row, display_text(texts[row]), color=INK_SECONDARY, fontsize=9, va='center', ha='right')
    _scale_legend(figure, _rect(figure, left, 0.38, 3.2, 0.12), DIVERGING, -3, 3,
                  ['-3 spreads', 'validation average', '+3 spreads'])

    # how alike every pair of rows is
    left += map_width + 1.1
    ax = _axes(figure, _rect(figure, left, bottom + map_height - matrix_size, matrix_size, matrix_size),
               grid_axis=None)
    ax.spines['bottom'].set_visible(False)
    ax.imshow(similarity, cmap=DIVERGING, vmin=-1, vmax=1, aspect='equal', interpolation='nearest')
    ax.set_xticks(np.arange(rows + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(rows + 1) - 0.5, minor=True)
    ax.grid(which='minor', color=SURFACE, linewidth=1)
    ax.tick_params(which='minor', length=0)
    for boundary in boundaries:
        ax.axhline(boundary, color=SURFACE, linewidth=5)
        ax.axvline(boundary, color=SURFACE, linewidth=5)
    centres = [np.flatnonzero(groups == group).mean() for group in np.unique(groups)]
    names = [str(group + 1) for group in np.unique(groups)]
    ax.set_xticks(centres)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_yticks(centres)
    ax.set_yticklabels(names, fontsize=9)
    ax.xaxis.tick_top()
    ax.set_title('Cosine similarity of every pair of rows, by font number', loc='left', fontsize=9,
                 color=INK_SECONDARY, pad=22)
    _scale_legend(figure, _rect(figure, left, bottom + map_height - matrix_size - 0.55, min(3.2, matrix_size), 0.12),
                  DIVERGING, -1, 1, ['-1 opposite', '0 unrelated', '1 same direction'])
    return figure
