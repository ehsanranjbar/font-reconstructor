"""
Turns a photo or a screenshot of text into the images the model was trained on.

The model expects what the capture simulation of the training data ends with: white text on black, cropped to
the text, one short piece of a line in each image. A real image is brought there in five steps:

  1. page          find the page, the largest evenly coloured area, and leave out what lies around it: the
                   desk beside a sheet of paper, the rest of a screen around a document
  2. cleanup       remove the background and its lighting, and stretch the contrast, so that the text is
                   bright and the paper is black whatever their colours were
  3. straightening a page that was photographed at an angle is pulled back into a rectangle, which also
                   makes its lines parallel again, and what tilt is left is turned out
  4. segmentation  find the lines of text, and cut every line at the gaps between words into pieces that are
                   about as wide as the texts the model was trained on
  5. framing       crop each piece to its text and fit it into the input size of the model

The steps are simple image statistics, no learned detector: they expect an image of text on a plain page, in
straight lines that are turned by less than about fifteen degrees.
"""
import collections
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

from font_reconstructor.dataset.capture import _otsu_threshold, _perspective_coefficients, _quantiles, fit_to_box

# the longest side of the image that is worked on. Larger images are scaled down, text of a few thousand
# pixels gains nothing over text of a hundred.
MAX_SIDE = 2400
# width of the blurred edge of a page that is left out, as a fraction of the longer side of the image
_PAGE_MARGIN = 0.004
# share of a piece of text that its strokes cover. Less is a scatter of specks, more a picture or a blot.
_MIN_COVER, _MAX_COVER = 0.04, 0.6


@dataclass
class Piece:
    """
    A piece of a line of text, ready for the model.

    :param image: uint8 array (height, width) of the model input, white text on black
    :param box: (left, top, right, bottom) of the piece in the cleaned, deskewed image of the page
    :param line: number of the line it was cut from, counted from 0 at the top
    """
    image: np.ndarray
    box: Tuple[int, int, int, int]
    line: int


def load_grey(image) -> np.ndarray:
    """
    :param image: a path or a PIL image, in any mode
    :return: float32 array (height, width) of brightness in [0, 255]
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    if image.mode in ('RGBA', 'LA', 'P'):
        # transparent pixels show the page behind them, which is white
        image = image.convert('RGBA')
        page = Image.new('RGBA', image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(page, image)
    image = image.convert('L')
    scale = MAX_SIDE / max(image.size)
    if scale < 1:
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)
    return np.asarray(image, dtype=np.float32)


def _largest_region(mask: np.ndarray) -> np.ndarray:
    """
    the largest connected area of True values of a small bool array, as a bool array of the same shape
    """
    height, width = mask.shape
    labels = np.zeros(mask.shape, dtype=np.int32)
    best, best_size, label = 0, 0, 0
    for start in zip(*np.nonzero(mask)):
        if labels[start]:
            continue
        label += 1
        labels[start] = label
        queue, size = collections.deque([start]), 0
        while queue:
            y, x = queue.popleft()
            size += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = label
                    queue.append((ny, nx))
        if size > best_size:
            best, best_size = label, size
    return labels == best if best else np.zeros(mask.shape, dtype=bool)


def find_page(pixels: np.ndarray, size: int = 200):
    """
    The page of an image: the largest evenly bright or evenly dark area that is roughly a rectangle.

    A scan or a screenshot is all page. A photo often shows more: the desk around a sheet of paper, or the
    rest of the screen around a document, with a brightness of its own that would be taken for a huge blot of
    ink. The area is looked for in a small copy of the image in which the strokes of the text are closed up.

    :param pixels: float array (height, width) of brightness in [0, 255]
    :return: (page, corners). page is a bool array like pixels, True for the page. corners are the four
             corners of the page as (x, y), from the top left clockwise. If the image shows nothing but the
             page, page is all True and corners is None.
    """
    height, width = pixels.shape
    scale = size / max(height, width)
    small_size = (max(8, round(width * scale)), max(8, round(height * scale)))
    small = np.asarray(Image.fromarray(pixels.astype(np.uint8)).resize(small_size, Image.BILINEAR)
                       .filter(ImageFilter.GaussianBlur(1)), dtype=np.float32)
    threshold = _otsu_threshold(small / 255.0) * 255.0

    best, best_score = None, 0.0
    for bright in (True, False):
        mask = Image.fromarray((((small > threshold) == bright) * 255).astype(np.uint8))
        # closing fills in the text of a page. Opening removes what is narrower than a page could be, such as
        # a reflection that touches it.
        closed = mask.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.MinFilter(9))
        opened = closed.filter(ImageFilter.MinFilter(25)).filter(ImageFilter.MaxFilter(25))
        region = _largest_region(np.asarray(opened) > 0)
        if not region.any():
            continue
        rows, columns = np.nonzero(region)
        box_area = (np.ptp(rows) + 1) * (np.ptp(columns) + 1)
        # large and filling its bounding box: a frame around a page is large too, but mostly hole
        score = region.mean() * (region.sum() / box_area) ** 2
        if score > best_score:
            best, best_score, page_is_bright = region, score, bright
    if best is None or best.mean() > 0.9:
        return np.ones(pixels.shape, dtype=bool), None

    # text and pictures on the page are holes in the area, which is filled from edge to edge both ways
    row_filled = np.maximum.accumulate(best, axis=1) & np.maximum.accumulate(best[:, ::-1], axis=1)[:, ::-1]
    column_filled = np.maximum.accumulate(best, axis=0) & np.maximum.accumulate(best[::-1], axis=0)[::-1]
    page = Image.fromarray(((row_filled & column_filled) * 255).astype(np.uint8))
    # the corners of a four-sided area are its points that lie furthest out along the two diagonals
    rows, columns = np.nonzero(np.asarray(page) > 0)
    picks = [np.argmin(columns + rows), np.argmax(columns - rows), np.argmax(columns + rows), np.argmin(columns - rows)]
    corners = np.array([(columns[pick] + 0.5, rows[pick] + 0.5) for pick in picks]) / scale
    mask = np.asarray(page.resize((width, height), Image.BILINEAR)) > 127
    # Only an area with four straight sides is a page seen at an angle. Anything else, such as a window with
    # a picture let into it, is used as it lies.
    x, y = corners[:, 0], corners[:, 1]
    four_sided = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    if not 0.93 < mask.sum() / max(four_sided, 1.0) < 1.07:
        return mask, None
    corners = _refine_corners(pixels, corners, threshold, page_is_bright, reach=3.0 / scale)
    return mask, corners.tolist()


def _refine_corners(pixels: np.ndarray, corners: np.ndarray, threshold: float, page_is_bright: bool,
                    reach: float) -> np.ndarray:
    """
    Find the edges of a page in the image itself, starting from corners that are only roughly right.

    Along every side, the brightness is followed from outside the page inwards at many points, up to `reach`
    pixels to either side of where the edge is thought to be. Where it crosses `threshold` is a point of the
    edge. A straight line is fitted to the points of a side, and the corners are where these lines meet.

    :param corners: array (4, 2) of (x, y), from the top left clockwise
    :return: array like corners
    """
    height, width = pixels.shape
    centre = corners.mean(axis=0)
    steps = np.arange(-reach, reach + 1.0)
    lines = []
    for side in range(4):
        start, end = corners[side], corners[(side + 1) % 4]
        along = (end - start) / max(float(np.linalg.norm(end - start)), 1e-9)
        inward = np.array([-along[1], along[0]])
        if np.dot(inward, centre - start) < 0:
            inward = -inward
        points = []
        # the ends of a side are left out, near a corner the neighbouring side gets in the way
        for share in np.linspace(0.12, 0.88, 40):
            base = start + share * (end - start)
            samples = base + steps[:, None] * inward
            x = np.clip(np.round(samples[:, 0]).astype(int), 0, width - 1)
            y = np.clip(np.round(samples[:, 1]).astype(int), 0, height - 1)
            on_page = (pixels[y, x] > threshold) == page_is_bright
            # the first point from which the next few are all page: the edge, not a speck outside it
            run = np.convolve(on_page, np.ones(4), mode='valid') == 4
            if run.any() and not on_page[0]:
                points.append(samples[int(np.argmax(run))])
        if len(points) < 8:
            lines.append((start, along))
            continue
        points = np.array(points)
        # distances from the side as it was thought to be. A line through their middle half is not thrown off
        # by the points that text or a reflection at the edge led astray.
        offsets = (points - start) @ inward
        low, high = np.quantile(offsets, (0.25, 0.75))
        kept = points[(offsets >= low - 2.0) & (offsets <= high + 2.0)]
        positions = (kept - start) @ along
        slope, intercept = np.polyfit(positions, (kept - start) @ inward, 1)
        origin = start + intercept * inward
        direction = along + slope * inward
        lines.append((origin, direction / np.linalg.norm(direction)))

    refined = []
    for side in range(4):
        (first_origin, first_direction), (second_origin, second_direction) = lines[side - 1], lines[side]
        system = np.array([first_direction, -second_direction]).T
        if abs(np.linalg.det(system)) < 1e-6:
            refined.append(corners[side])
            continue
        first_step, _ = np.linalg.solve(system, second_origin - first_origin)
        refined.append(first_origin + first_step * first_direction)
    return np.array(refined)


def straighten(pixels: np.ndarray, corners):
    """
    Pull a page that was photographed at an angle back into a rectangle.

    The lines of a page run along its upper and lower edge. Seen at an angle these edges are not parallel,
    and neither are the lines, so no turning of the image makes all of them horizontal. Mapping the four
    corners of the page to those of a rectangle does.

    :param corners: the corners of the page as (x, y), from the top left clockwise, see `find_page`
    :return: (pixels, page) of the rectangle: its brightness and the area of it that is page
    """
    points = np.asarray(corners, dtype=np.float64)

    def length(first, second):
        return float(np.linalg.norm(points[first] - points[second]))

    width = max(8, round((length(0, 1) + length(3, 2)) / 2))
    height = max(8, round((length(0, 3) + length(1, 2)) / 2))
    rectangle = [(0, 0), (width, 0), (width, height), (0, height)]
    coefficients = _perspective_coefficients(rectangle, points.tolist())
    straight = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)).transform(
        (width, height), Image.PERSPECTIVE, coefficients, Image.BICUBIC)
    # The step in brightness at the edge of the page is no text. It is a few pixels wide, a narrow frame of
    # the rectangle is left out for it.
    frame = max(2, round(_PAGE_MARGIN * max(pixels.shape)))
    page = np.zeros((height, width), dtype=bool)
    page[frame:height - frame, frame:width - frame] = True
    return np.asarray(straight, dtype=np.float32), page


def local_background(pixels: np.ndarray, page: np.ndarray, cells: int = 60) -> np.ndarray:
    """
    The brightness of the paper at every pixel, for dark text on a brighter page.

    Photos are not lit evenly: there are shadows, the corners of a lens are darker, a screen shows a pattern.
    The paper level is therefore read locally. The image is divided into cells, about `cells` along its
    shorter side, and the brightest pixel of each cell and its neighbours is taken for paper. That holds as
    long as every such neighbourhood shows some paper, that is for strokes thinner than about a sixth of the
    shorter side of the image.

    :param page: bool array like pixels, only this area is looked at
    :return: float32 array like pixels
    """
    height, width = pixels.shape
    cell = max(1, int(np.ceil(min(height, width) / cells)))
    rows, columns = -(-height // cell), -(-width // cell)
    padded = np.zeros((rows * cell, columns * cell), dtype=np.float32)
    padded[:height, :width] = np.where(page, pixels, 0.0)
    brightest = padded.reshape(rows, cell, columns, cell).max(axis=(1, 3))
    covered = brightest > 0

    brightest = Image.fromarray(brightest.astype(np.uint8)).filter(ImageFilter.MaxFilter(9))
    # smoothed over the cells that lie on the page only, so that what is outside does not darken its edge
    weight = Image.fromarray((covered * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(9))
    level = np.asarray(brightest.filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)
    weight = np.asarray(weight.filter(ImageFilter.GaussianBlur(2)), dtype=np.float32) / 255.0
    level = level / np.maximum(weight, 1e-3)
    full = Image.fromarray(np.clip(level, 0, 255).astype(np.uint8)).resize((columns * cell, rows * cell), Image.BILINEAR)
    return np.asarray(full, dtype=np.float32)[:height, :width]


def clean(pixels: np.ndarray, page: Optional[np.ndarray] = None) -> np.ndarray:
    """
    The ink of an image: 0 for the background, 1 for the text, whatever their brightness and under uneven light.

    The background is what most of the page shows, so text that is brighter than it is inverted first. Its
    level is then read from the brightest spots of every neighbourhood, see `local_background`, and the
    contrast is stretched from just above the noise of the background to the darkest strokes.

    :param pixels: float array (height, width) of brightness in [0, 255]
    :param page: bool array like pixels of the area to look at, see `find_page`. What lies outside it comes
        out as background.
    :return: float32 array like pixels, in [0, 1]
    """
    if page is None:
        page = np.ones(pixels.shape, dtype=bool)
    inside = pixels[page]
    low, middle, high = _quantiles(inside, (0.02, 0.5, 0.98))
    if middle - low < high - middle:
        # a dark background with bright text
        pixels = 255.0 - pixels
        middle = 255.0 - middle

    background = local_background(pixels, page)
    ink = np.where(page, (background - pixels) / np.maximum(background, 1.0), 0.0)

    floor_low, level = _quantiles(ink[page], (0.05, 0.30))
    noise = max(float(level - floor_low), 0.0) / 1.1
    smoothed = Image.fromarray((np.clip(ink, 0.0, 1.0) * 255).astype(np.uint8)).filter(ImageFilter.BoxBlur(1))
    peak = max(float(_quantiles(np.asarray(smoothed, dtype=np.float32), (0.999,))[0]) / 255.0, 0.05)
    floor = min(float(level) + 2.0 * noise, 0.4 * peak)
    ink = np.clip((ink - floor) / (peak - floor), 0.0, 1.0).astype(np.float32)
    if not page.all():
        # the edge of a page is a step in brightness that is no text, a narrow band along it is left blank
        band = 2 * max(1, round(0.012 * min(pixels.shape))) + 1
        inner = np.asarray(Image.fromarray((page * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(band))) > 0
        ink = np.where(inner, ink, 0.0).astype(np.float32)
    return ink


def estimate_skew(ink: np.ndarray, limit: float = 15.0, size: int = 400) -> float:
    """
    The angle by which the lines of text are turned, in degrees, positive for lines that rise to the right.

    Rows that run along the lines are full of ink or empty, with a sharp step from one to the other at the
    top and bottom of every line. Rows that cross the lines all hold a little. So the image is turned by every
    angle up to `limit`, and the one that makes neighbouring rows differ most is taken.
    """
    height, width = ink.shape
    scale = min(1.0, size / max(height, width))
    small = Image.fromarray((ink * 255).astype(np.uint8)).resize(
        (max(8, round(width * scale)), max(8, round(height * scale))), Image.BILINEAR)
    if not np.asarray(small).any():
        return 0.0

    def sharpness(angle):
        rows = np.asarray(small.rotate(-angle, Image.BILINEAR), dtype=np.float32).sum(axis=1)
        return float(np.square(np.diff(rows)).sum())

    best = max(np.arange(-limit, limit + 0.5, 1.0), key=sharpness)
    return float(max(np.arange(best - 0.75, best + 1.0, 0.25), key=sharpness))


def deskew(ink: np.ndarray, angle: float) -> np.ndarray:
    """
    turn a cleaned image so that lines of text that are turned by `angle` run horizontally
    """
    if abs(angle) < 0.25:
        return ink
    turned = Image.fromarray((ink * 255).astype(np.uint8)).rotate(-angle, Image.BICUBIC, expand=True)
    return np.asarray(turned, dtype=np.float32) / 255.0


def strokes(ink: np.ndarray) -> np.ndarray:
    """
    Where a cleaned image shows strokes of text, as a bool array.

    Dust, sensor noise and the dots of a screen leave bright specks of a pixel or two, which a median over
    three by three pixels removes. Strokes are wider than that.
    """
    smooth = np.asarray(Image.fromarray((ink * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(3)),
                        dtype=np.float32) / 255.0
    return smooth > max(_otsu_threshold(smooth), 0.15)


def _runs(present: np.ndarray) -> List[Tuple[int, int]]:
    """
    the stretches of True values of a 1d array, as (start, stop) with stop exclusive
    """
    padded = np.concatenate([[False], present, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return [(int(start), int(stop)) for start, stop in zip(edges[::2], edges[1::2])]


def find_lines(ink: np.ndarray, min_height: int = 6) -> List[Tuple[int, int]]:
    """
    The lines of text of a cleaned image, as (top, bottom) rows with bottom exclusive.

    A row belongs to a line if it shows ink. Dots and marks above and below the letters leave thin empty rows
    inside a line, so a flat stretch of rows that lies close to a tall one is joined to it. Two tall stretches
    are two lines however close they are.
    """
    rows = strokes(ink).sum(axis=1)
    if not rows.any():
        return []
    # A row of a line holds a fair share of the ink of a typical row with ink, a row between lines only
    # specks. The typical row is the median one, so that a few heavy lines do not hide the light ones.
    rows = np.convolve(rows, np.ones(3) / 3, mode='same')
    runs = _runs(rows >= max(1.0, 0.15 * float(np.median(rows[rows > 0]))))

    merged = [runs[0]]
    for top, bottom in runs[1:]:
        last_top, last_bottom = merged[-1]
        tall, flat = sorted((bottom - top, last_bottom - last_top), reverse=True)
        if top - last_bottom < 0.25 * tall and flat < 0.4 * tall:
            merged[-1] = (last_top, bottom)
        else:
            merged.append((top, bottom))
    runs = merged
    # A stretch much flatter than the lines is a row of dots or marks that stands apart from its line. It goes
    # to the nearest line if that is close, and is left out otherwise: a speck, or a rule across the page.
    typical = float(np.median([bottom - top for top, bottom in runs]))
    lines = [[top, bottom] for top, bottom in runs if bottom - top >= 0.4 * typical]
    for top, bottom in runs:
        if bottom - top >= 0.4 * typical or not lines:
            continue
        distances = [max(line[0] - bottom, top - line[1], 0) for line in lines]
        nearest = int(np.argmin(distances))
        if distances[nearest] <= 0.5 * typical:
            lines[nearest] = [min(lines[nearest][0], top), max(lines[nearest][1], bottom)]
    return [(top, bottom) for top, bottom in lines if bottom - top >= min_height]


def find_pieces(line: np.ndarray, max_aspect: float = 6.5, word_gap: float = 0.3) -> List[Tuple[int, int]]:
    """
    Cut a line of text at the gaps between its words into pieces, as (left, right) columns with right exclusive.

    The model was trained on short texts in an input four times as wide as it is high. A whole line squeezed
    into that would show letters far smaller than any it has seen, so the words are gathered into pieces that
    are at most `max_aspect` times as wide as the line is high, which is about what the longest training texts
    are. A single word that is wider stays whole. What is much lower than the line, a speck or a mark beside
    the text, is left out.

    :param line: cleaned image of one line
    :param word_gap: narrowest gap that separates two words, as a fraction of the line height. Letters of one
        word that do not join stand closer together than that.
    """
    binary = strokes(line)
    rows = np.flatnonzero(binary.any(axis=1))
    if len(rows) == 0:
        return []
    height = int(rows[-1] - rows[0] + 1)
    words = []
    for left, right in _runs(binary.sum(axis=0) > 0):
        rows = np.flatnonzero(binary[:, left:right].any(axis=1))
        if rows[-1] - rows[0] + 1 >= 0.3 * height:
            words.append((left, right))
    if not words:
        return []
    joined = [list(words[0])]
    for left, right in words[1:]:
        if left - joined[-1][1] < word_gap * height:
            joined[-1][1] = right
        else:
            joined.append([left, right])

    pieces = [joined[0]]
    for left, right in joined[1:]:
        if right - pieces[-1][0] <= max_aspect * height:
            pieces[-1][1] = right
        else:
            pieces.append([left, right])
    # a last piece that is only a scrap, a single short word, is more use together with the one before it
    if len(pieces) > 1 and pieces[-1][1] - pieces[-1][0] < 1.2 * height:
        if pieces[-1][1] - pieces[-2][0] <= 1.5 * max_aspect * height:
            scrap = pieces.pop()
            pieces[-1][1] = scrap[1]
    # Text that is set tightly has no gap wide enough to count as one between words. Such a piece is cut
    # anyway, at the gap between two letters that lies nearest to where it should end.
    ink_columns = binary.any(axis=0)
    cut = []
    for left, right in pieces:
        parts = int(np.ceil((right - left) / (1.5 * max_aspect * height)))
        edges = [left]
        for part in range(1, parts):
            wanted = left + part * (right - left) / parts
            gaps = [column for column in range(edges[-1] + height, right - height) if not ink_columns[column]]
            edges.append(min(gaps, key=lambda column: abs(column - wanted)) if gaps else int(wanted))
        edges.append(right)
        cut.extend(zip(edges[:-1], edges[1:]))
    # a piece is a word at least. What is much narrower than the line is high is a stray stroke.
    return [(int(left), int(right)) for left, right in cut if right - left >= 0.5 * height]


def clean_page(image) -> np.ndarray:
    """
    Steps one to three: the page of an image, straightened, as ink on a black background.

    :param image: a path or a PIL image
    :return: float32 array (height, width) in [0, 1], 1 for the text
    """
    pixels = load_grey(image)
    page, corners = find_page(pixels)
    if corners is not None:
        pixels, page = straighten(pixels, corners)
    elif not page.all():
        rows, columns = np.nonzero(page)
        box = (slice(rows.min(), rows.max() + 1), slice(columns.min(), columns.max() + 1))
        pixels, page = pixels[box], page[box]
    ink = clean(pixels, page)
    return deskew(ink, estimate_skew(ink))


def prepare(image, dims: Tuple[int, int] = (256, 64), margin: float = 0.1, max_pieces: int = 64) -> List[Piece]:
    """
    All of the above: the pieces of text of an image, each as an input of the model.

    :param image: a path or a PIL image
    :param dims: (width, height) of the model input
    :param margin: background kept around the text of a piece, as a fraction of its height. The training
        images have between none and a quarter.
    :param max_pieces: at most this many pieces are returned, the largest ones first in reading order
    :return: the pieces, from the top line to the bottom one and from left to right within a line. Empty if
             the image shows no text.
    """
    ink = clean_page(image)
    found = []
    for number, (top, bottom) in enumerate(find_lines(ink)):
        line = ink[top:bottom]
        for left, right in find_pieces(line):
            found.append((number, (left, top, right, bottom)))
    if len(found) > max_pieces:
        largest = sorted(range(len(found)), key=lambda i: -(found[i][1][2] - found[i][1][0]) * (found[i][1][3] - found[i][1][1]))
        found = [found[i] for i in sorted(largest[:max_pieces])]

    pieces = []
    for number, (left, top, right, bottom) in found:
        crop = ink[top:bottom, left:right]
        # text covers a fair part of its box, a few specks that were taken for a word do not
        if not _MIN_COVER < strokes(crop).mean() < _MAX_COVER:
            continue
        fitted = np.asarray(fit_to_box(Image.fromarray((crop * 255).astype(np.uint8)), dims, margins=(margin,) * 4))
        if fitted.max() > 0:
            pieces.append(Piece(fitted, (left, top, right, bottom), number))
    return pieces
