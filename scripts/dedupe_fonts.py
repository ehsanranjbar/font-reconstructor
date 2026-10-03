"""
Find and remove near-duplicate fonts from the font collection.

Some typefaces appear several times in the collection under different names (different foundry packs, small metric
differences) without being byte-identical. For training, such clones are harmful: the model sees them as different
classes with the same appearance. Different weights or styles of a typeface (Regular, Bold, Italic, Outline) are not
duplicates and must be kept.

Method
------
1. Fingerprint: every character of a font's ``supported_charset`` is drawn alone with Pillow (64 px, white on black),
   cropped to its ink bounding box, scaled to fit a 32x32 box keeping the aspect ratio and centred on black. This is
   the normalisation of the model's training target, so two fonts with the same fingerprint are the same class to the
   model. The result is a (glyphs, 32, 32) uint8 array per font.
2. Distances: every pair of fonts is compared glyph by glyph. For a glyph the distance is the mean absolute pixel
   difference on a 0..1 scale. A pair gets two numbers: ``mean`` (average over all glyphs) and ``worst`` (the largest
   single-glyph distance). Distances are computed exactly, in chunks, so memory stays small.
3. Duplicate pairs: a pair is a duplicate only if ``mean <= --threshold`` and ``worst <= --glyph-threshold``, so one
   very different glyph (for example an alternative dot or a different digit) keeps two fonts apart.
4. Groups: duplicate pairs are merged by complete linkage. Two clusters are only merged if every cross pair of fonts is
   a duplicate pair, so A~B and B~C never merge A and C when A and C differ.
5. Keep rule: in each group the font that is kept is, in this order, an openly licensed font (manifest licence not
   starting with ``unknown``), then the most central one (smallest mean distance to the other members), then the one
   with the shortest file name (ties are broken alphabetically).

Usage
-----
Dry run (default, changes nothing, writes reports to ``--report-dir``)::

    python scripts/dedupe_fonts.py
    python scripts/dedupe_fonts.py --threshold 0.01 --glyph-threshold 0.05
    python scripts/dedupe_fonts.py --sweep 0.005 0.01 0.02:0.1   # removed counts for other thresholds

Reports: ``duplicates.csv`` (one row per font that would be removed), ``nearest_neighbours.csv`` (nearest other font of
every font, for choosing the threshold), ``pairs.png`` (contact sheet of pairs across the distance range, each pair
shows the two fingerprints and the pixel difference in red) and ``skipped_fonts.csv`` (fonts that could not be used).

Removal, reversible::

    python scripts/dedupe_fonts.py --apply

moves the duplicate font files to ``--duplicates-dir`` (never deletes them), copies ``fonts.csv`` and the manifest to
timestamped ``.bak`` files next to them and rewrites both without the removed rows, keeping the columns and the row
order. To undo, move the files back and restore the ``.bak`` files.

Only the standard library, numpy and Pillow are used.
"""
import argparse
import csv
import hashlib
import io
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GLYPH_SIZE = 32  # side of the fingerprint of one glyph, as in the training target
RENDER_SIZE = 64  # font size used to draw a glyph before it is scaled to GLYPH_SIZE
DEFAULT_THRESHOLD = 0.01
DEFAULT_GLYPH_THRESHOLD = 0.05
UNKNOWN_LICENSE_PREFIX = "unknown"
CHUNK_ROWS = 256  # fonts compared against one font at a time, bounds the size of temporary arrays
SHEET_COLUMNS = 3
GRID_COLUMNS = 14  # glyphs per row in the contact sheet


# ----------------------------------------------------------------------------------------------------------------------
# reading the collection
# ----------------------------------------------------------------------------------------------------------------------

def read_csv(path: str) -> Tuple[List[str], List[Dict[str, str]], str]:
    """
    Read a csv file.

    :return: the column names, the rows as dicts and the line terminator the file uses
    """
    with open(path, newline="", encoding="utf-8") as f:
        raw = f.read()  # newline="" keeps the original line terminators
    terminator = "\r\n" if "\r\n" in raw else "\n"
    reader = csv.DictReader(io.StringIO(raw, newline=""))
    rows = list(reader)
    return list(reader.fieldnames or []), rows, terminator


def write_csv(path: str, columns: Sequence[str], rows: Sequence[Dict[str, str]], terminator: str = "\n") -> None:
    """
    Write rows to a csv file, through a temporary file so a failure never leaves a half written file.
    """
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), lineterminator=terminator, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def is_open_license(license_text: Optional[str]) -> bool:
    """
    Whether a manifest licence value names a real licence. Empty and ``unknown...`` values do not.
    """
    value = (license_text or "").strip().lower()
    return bool(value) and not value.startswith(UNKNOWN_LICENSE_PREFIX)


# ----------------------------------------------------------------------------------------------------------------------
# fingerprints
# ----------------------------------------------------------------------------------------------------------------------

def render_glyph(font: ImageFont.FreeTypeFont, char: str) -> np.ndarray:
    """
    Draw one character, crop it to its ink and fit it into a GLYPH_SIZE square, centred, keeping the aspect ratio.

    :return: uint8 array (GLYPH_SIZE, GLYPH_SIZE), black if the font has no ink for the character
    """
    out = np.zeros((GLYPH_SIZE, GLYPH_SIZE), dtype=np.uint8)
    left, top, right, bottom = font.getbbox(char, anchor="lt")
    if right <= left or bottom <= top:
        return out
    # the box may start left of or above the origin for glyphs with negative bearings
    shift_x, shift_y = -min(left, 0), -min(top, 0)
    canvas = Image.new("L", (right + shift_x, bottom + shift_y), 0)
    ImageDraw.Draw(canvas).text((shift_x, shift_y), char, fill=255, anchor="lt", font=font)
    ink = canvas.getbbox()
    if ink is None:
        return out
    glyph = canvas.crop(ink)
    width, height = glyph.size
    scale = min(GLYPH_SIZE / width, GLYPH_SIZE / height)
    new_size = (max(1, min(GLYPH_SIZE, round(width * scale))), max(1, min(GLYPH_SIZE, round(height * scale))))
    glyph = glyph.resize(new_size, Image.LANCZOS)
    box = ((GLYPH_SIZE - new_size[0]) // 2, (GLYPH_SIZE - new_size[1]) // 2)
    page = Image.new("L", (GLYPH_SIZE, GLYPH_SIZE), 0)
    page.paste(glyph, box)
    return np.asarray(page, dtype=np.uint8)


def render_fingerprint(job: Tuple[str, str]) -> Tuple[Optional[np.ndarray], str]:
    """
    Render the fingerprint of one font. Runs in a worker process.

    :param job: (font path, characters)
    :return: (array (len(characters), GLYPH_SIZE, GLYPH_SIZE) or None, error message, empty on success)
    """
    path, charset = job
    try:
        font = ImageFont.truetype(path, RENDER_SIZE, encoding="unic")
        glyphs = [render_glyph(font, char) for char in charset]
    except Exception as e:  # any failure to open or draw means the font is unusable
        return None, f"{type(e).__name__}: {e}"
    return np.stack(glyphs), ""


def collection_key(rows: Sequence[Dict[str, str]], fonts_dir: str) -> str:
    """
    A hash of everything the fingerprints depend on, so a cache is only reused for the same files and settings.
    """
    h = hashlib.sha256(f"{GLYPH_SIZE}/{RENDER_SIZE}".encode())
    for row in rows:
        path = os.path.join(fonts_dir, row["file"])
        try:
            stat = os.stat(path)
            h.update(f"{row['file']}|{stat.st_size}|{stat.st_mtime_ns}|{row['supported_charset']}\n".encode())
        except OSError:
            h.update(f"{row['file']}|missing\n".encode())
    return h.hexdigest()


def compute_fingerprints(
    rows: Sequence[Dict[str, str]], fonts_dir: str, workers: int, cache_path: Optional[str]
) -> Tuple[np.ndarray, List[int], List[Tuple[str, str]]]:
    """
    Fingerprint every font of the annotation rows.

    :param workers: number of worker processes
    :param cache_path: npz file to reuse fingerprints from and store them to, or None
    :return: (fingerprints (usable fonts, glyphs, GLYPH_SIZE, GLYPH_SIZE) uint8, indices of the usable rows,
        list of (file, reason) for fonts that were skipped)
    """
    key = collection_key(rows, fonts_dir)
    if cache_path and os.path.exists(cache_path):
        try:
            with np.load(cache_path) as cached:
                if str(cached["key"]) == key:
                    print(f"Using cached fingerprints from {cache_path}")
                    skipped_cached = [tuple(item) for item in cached["skipped"].reshape(-1, 2).tolist()]
                    return cached["prints"], cached["usable"].tolist(), skipped_cached
        except (OSError, KeyError, ValueError):
            pass  # unreadable cache, compute again

    num_glyphs = max(len(row["supported_charset"]) for row in rows)
    jobs = [(os.path.join(fonts_dir, row["file"]), row["supported_charset"]) for row in rows]
    prints, usable, skipped = [], [], []
    start = time.time()
    with Pool(processes=workers) as pool:
        for i, (print_, error) in enumerate(pool.imap(render_fingerprint, jobs, chunksize=4)):
            if print_ is None:
                skipped.append((rows[i]["file"], error))
            else:
                padded = np.zeros((num_glyphs, GLYPH_SIZE, GLYPH_SIZE), dtype=np.uint8)
                padded[: len(print_)] = print_
                prints.append(padded)
                usable.append(i)
            if (i + 1) % 100 == 0:
                print(f"  rendered {i + 1}/{len(jobs)} fonts ({time.time() - start:.0f}s)", flush=True)
    stacked = np.stack(prints) if prints else np.zeros((0, num_glyphs, GLYPH_SIZE, GLYPH_SIZE), dtype=np.uint8)
    if cache_path:
        np.savez(cache_path, key=np.array(key), prints=stacked, usable=np.array(usable, dtype=np.int64),
                 skipped=np.array(skipped, dtype=str).reshape(-1, 2))
    return stacked, usable, skipped


# ----------------------------------------------------------------------------------------------------------------------
# distances
# ----------------------------------------------------------------------------------------------------------------------

def _distances_to_later_fonts(flat: np.ndarray, i: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Distances of font i to the fonts after it: mean over glyphs and largest glyph distance, both on a 0..1 scale.
    """
    n, _, size = flat.shape
    means = np.empty(n - i - 1, dtype=np.float32)
    worst = np.empty(n - i - 1, dtype=np.float32)
    target = flat[i]
    for start in range(i + 1, n, CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, n)
        block = flat[start:stop]
        # |a - b| == max(a, b) - min(a, b), exact and without leaving uint8
        diff = np.maximum(block, target) - np.minimum(block, target)
        per_glyph = diff.sum(axis=2, dtype=np.uint32).astype(np.float32) / (255.0 * size)
        means[start - i - 1: stop - i - 1] = per_glyph.mean(axis=1)
        worst[start - i - 1: stop - i - 1] = per_glyph.max(axis=1)
    return means, worst


def pairwise_distances(prints: np.ndarray, workers: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compare all pairs of fonts exactly, in chunks.

    :param prints: fingerprints (fonts, glyphs, H, W) uint8
    :param workers: number of threads
    :return: two symmetric (fonts, fonts) float32 matrices, the mean glyph distance and the largest glyph distance.
        The diagonal is infinite.
    """
    n = len(prints)
    flat = prints.reshape(n, prints.shape[1], -1)
    mean_d = np.full((n, n), np.inf, dtype=np.float32)
    worst_d = np.full((n, n), np.inf, dtype=np.float32)
    start = time.time()

    def work(i: int) -> int:
        means, worst = _distances_to_later_fonts(flat, i)
        mean_d[i, i + 1:] = means
        mean_d[i + 1:, i] = means
        worst_d[i, i + 1:] = worst
        worst_d[i + 1:, i] = worst
        return i

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, _ in enumerate(pool.map(work, range(n - 1))):
            if (done + 1) % 200 == 0:
                print(f"  compared {done + 1}/{n} fonts ({time.time() - start:.0f}s)", flush=True)
    return mean_d, worst_d


# ----------------------------------------------------------------------------------------------------------------------
# grouping
# ----------------------------------------------------------------------------------------------------------------------

def group_duplicates(mean_d: np.ndarray, worst_d: np.ndarray, threshold: float, glyph_threshold: float) -> List[List[int]]:
    """
    Group duplicate fonts by complete linkage.

    A pair is a duplicate pair if mean <= threshold and worst <= glyph_threshold. Pairs are visited from the closest
    to the farthest, and two clusters are merged only if every cross pair is a duplicate pair, so every member of a
    group is within the thresholds of every other member.

    :return: groups of font indices with at least two members, members sorted by index
    """
    adjacent = (mean_d <= threshold) & (worst_d <= glyph_threshold)
    pairs = np.argwhere(np.triu(adjacent, 1))
    order = np.argsort(mean_d[pairs[:, 0], pairs[:, 1]], kind="stable")
    members: Dict[int, List[int]] = {}
    label = np.arange(len(mean_d))
    for i, j in pairs[order]:
        a, b = label[i], label[j]
        if a == b:
            continue
        group_a, group_b = members.get(a, [a]), members.get(b, [b])
        if adjacent[np.ix_(group_a, group_b)].all():
            merged = group_a + group_b
            members.pop(b, None)
            members[a] = merged
            label[merged] = a
    return [sorted(group) for group in members.values() if len(group) > 1]


def choose_keeper(group: Sequence[int], mean_d: np.ndarray, files: Sequence[str], open_license: Sequence[bool]) -> int:
    """
    Pick the font to keep from a group: an openly licensed font first, then the most central one (smallest mean
    distance to the other members), then the shortest file name, then the alphabetically first file name.
    """
    candidates = [i for i in group if open_license[i]] or list(group)
    centrality = {i: float(np.mean([mean_d[i, j] for j in group if j != i])) for i in candidates}
    best = min(centrality.values())
    # distances are float32 sums, so ties between equally central fonts are compared with a small tolerance
    central = [i for i in candidates if centrality[i] <= best + 1e-7]
    return min(central, key=lambda i: (len(files[i]), files[i]))


def plan_removals(
    mean_d: np.ndarray, worst_d: np.ndarray, threshold: float, glyph_threshold: float,
    files: Sequence[str], open_license: Sequence[bool],
) -> List[Tuple[int, List[int]]]:
    """
    Group the duplicates and choose the font to keep in each group.

    :return: list of (kept index, indices of the removed fonts), largest groups first
    """
    plan = []
    for group in group_duplicates(mean_d, worst_d, threshold, glyph_threshold):
        keeper = choose_keeper(group, mean_d, files, open_license)
        plan.append((keeper, [i for i in group if i != keeper]))
    plan.sort(key=lambda item: (-len(item[1]), files[item[0]]))
    return plan


# ----------------------------------------------------------------------------------------------------------------------
# reports
# ----------------------------------------------------------------------------------------------------------------------

def write_duplicates_csv(
    path: str, plan: Sequence[Tuple[int, List[int]]], mean_d: np.ndarray, worst_d: np.ndarray,
    info: Sequence[Dict[str, str]],
) -> None:
    """
    One row per font that would be removed, with its kept font and its distances to it.
    """
    columns = ["group", "group_size", "kept_file", "kept_font", "kept_license", "removed_file", "removed_font",
               "removed_family", "removed_license", "mean_distance", "max_glyph_distance"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for number, (keeper, removed) in enumerate(plan, 1):
            for i in removed:
                writer.writerow([
                    number, len(removed) + 1, info[keeper]["file"], info[keeper]["font"], info[keeper]["license"],
                    info[i]["file"], info[i]["font"], info[i]["family"], info[i]["license"],
                    f"{mean_d[keeper, i]:.5f}", f"{worst_d[keeper, i]:.5f}",
                ])


def nearest_neighbours(mean_d: np.ndarray) -> np.ndarray:
    """
    Index of the nearest other font (smallest mean distance) of every font.
    """
    return np.argmin(mean_d, axis=1)


def write_nearest_csv(
    path: str, mean_d: np.ndarray, worst_d: np.ndarray, info: Sequence[Dict[str, str]], removed: set
) -> None:
    """
    For every font: its nearest other font and the distances to it, to choose the threshold from.
    """
    nearest = nearest_neighbours(mean_d)
    columns = ["file", "font", "family", "nearest_file", "nearest_font", "nearest_family", "mean_distance",
               "max_glyph_distance", "same_family", "would_be_removed"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        order = np.argsort(mean_d[np.arange(len(nearest)), nearest], kind="stable")
        for i in order:
            j = int(nearest[i])
            writer.writerow([
                info[i]["file"], info[i]["font"], info[i]["family"], info[j]["file"], info[j]["font"],
                info[j]["family"], f"{mean_d[i, j]:.5f}", f"{worst_d[i, j]:.5f}",
                int(info[i]["family"] == info[j]["family"]), int(int(i) in removed),
            ])


def select_sheet_pairs(mean_d: np.ndarray, threshold: float, count: int) -> List[Tuple[int, int]]:
    """
    Choose pairs for the contact sheet: nearest-neighbour pairs spread over the whole distance range, plus pairs just
    below and just above the threshold, so a person can judge where duplicates end. Sorted by distance.
    """
    nearest = nearest_neighbours(mean_d)
    unique = {(min(i, int(j)), max(i, int(j))) for i, j in enumerate(nearest)}
    pairs = sorted(unique, key=lambda p: mean_d[p])
    if len(pairs) <= count:
        return pairs
    chosen = {pairs[int(k)] for k in np.linspace(0, len(pairs) - 1, count // 2)}
    values = np.array([mean_d[p] for p in pairs])
    split = int(np.searchsorted(values, threshold))
    near = count - len(chosen)
    below = pairs[max(0, split - near // 2): split]
    above = pairs[split: split + (near - len(below))]
    chosen.update(below)
    chosen.update(above)
    return sorted(chosen, key=lambda p: mean_d[p])


def _ascii(text: str, limit: int = 44) -> str:
    """
    A label that the default Pillow bitmap font can draw.
    """
    return text.encode("ascii", "replace").decode()[:limit]


def _glyph_grid(prints: np.ndarray, red_only: bool = False, columns: int = GRID_COLUMNS) -> Image.Image:
    """
    Lay the glyphs of a fingerprint out in a grid with one pixel grey separators.

    :param red_only: draw the glyphs in red on black instead of white on black
    """
    num = len(prints)
    rows = -(-num // columns)
    cell = GLYPH_SIZE + 1
    grid = np.full((rows * cell + 1, columns * cell + 1, 3), 70, dtype=np.uint8)
    for k in range(num):
        r, c = divmod(k, columns)
        tile = np.repeat(prints[k][..., None], 3, axis=2)
        if red_only:
            tile[..., 1:] = 0
        grid[1 + r * cell: 1 + r * cell + GLYPH_SIZE, 1 + c * cell: 1 + c * cell + GLYPH_SIZE] = tile
    return Image.fromarray(grid)


def write_pairs_png(
    path: str, pairs: Sequence[Tuple[int, int]], prints: np.ndarray, mean_d: np.ndarray, worst_d: np.ndarray,
    info: Sequence[Dict[str, str]], threshold: float,
) -> None:
    """
    Contact sheet. Each block shows the two fingerprints one above the other and below them their difference in red.
    Blocks are ordered by distance (left to right, top to bottom). Blocks of pairs the script would treat as
    duplicates have a green header, the others a grey one.
    """
    font = ImageFont.load_default()
    grid_w, grid_h = _glyph_grid(prints[0]).size
    line = 14
    block_w, block_h = grid_w + 12, 4 * line + 3 * grid_h + 12
    rows = -(-len(pairs) // SHEET_COLUMNS)
    sheet = Image.new("RGB", (SHEET_COLUMNS * block_w, rows * block_h), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    for number, (i, j) in enumerate(pairs):
        r, c = divmod(number, SHEET_COLUMNS)
        x0, y0 = c * block_w + 6, r * block_h + 4
        duplicate = mean_d[i, j] <= threshold
        draw.rectangle([x0 - 3, y0 - 2, x0 + grid_w + 2, y0 + line - 2], fill=(30, 110, 50) if duplicate else (80, 80, 80))
        draw.text((x0, y0), f"#{number + 1}  mean {mean_d[i, j]:.4f}  worst glyph {worst_d[i, j]:.3f}", fill="white", font=font)
        draw.text((x0, y0 + line), _ascii(info[i]["file"]), fill=(255, 230, 120), font=font)
        sheet.paste(_glyph_grid(prints[i]), (x0, y0 + 2 * line))
        draw.text((x0, y0 + 2 * line + grid_h + 2), _ascii(info[j]["file"]), fill=(130, 220, 255), font=font)
        sheet.paste(_glyph_grid(prints[j]), (x0, y0 + 3 * line + grid_h))
        sheet.paste(_glyph_grid(np.maximum(prints[i], prints[j]) - np.minimum(prints[i], prints[j]), red_only=True), (x0, y0 + 3 * line + 2 * grid_h + 4))
    sheet.save(path)


# ----------------------------------------------------------------------------------------------------------------------
# applying
# ----------------------------------------------------------------------------------------------------------------------

def apply_removals(
    removed_files: Sequence[str], fonts_dir: str, duplicates_dir: str, annotations: str, manifest: Optional[str]
) -> None:
    """
    Remove duplicates reversibly: back up both csv files with a timestamp, move the font files to duplicates_dir (they
    are never deleted) and rewrite both csv files without the removed rows, keeping the columns and the row order.
    If a file can not be moved, everything moved so far is moved back and the csv files are left untouched.
    """
    removed = set(removed_files)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    csv_paths = [annotations] + ([manifest] if manifest else [])
    new_tables = []
    for csv_path in csv_paths:
        columns, rows, terminator = read_csv(csv_path)
        kept_rows = [row for row in rows if row["file"] not in removed]
        new_tables.append((csv_path, columns, kept_rows, terminator, len(rows)))

    os.makedirs(duplicates_dir, exist_ok=True)
    moves: List[Tuple[str, str]] = []
    try:
        for name in removed_files:
            source = os.path.join(fonts_dir, name)
            target = os.path.join(duplicates_dir, name)
            if os.path.exists(target):
                raise FileExistsError(f"{target} already exists, refusing to overwrite it")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.move(source, target)
            moves.append((source, target))
    except Exception:
        for source, target in reversed(moves):
            shutil.move(target, source)
        print(f"ERROR: moving the fonts failed, the {len(moves)} fonts moved so far were moved back, "
              f"csv files untouched.", file=sys.stderr)
        raise

    for csv_path, columns, kept_rows, terminator, before in new_tables:
        backup = f"{csv_path}.{stamp}.bak"
        shutil.copy2(csv_path, backup)
        write_csv(csv_path, columns, kept_rows, terminator)
        print(f"{csv_path}: {before} -> {len(kept_rows)} rows (backup: {backup})")
    print(f"Moved {len(moves)} font files from {fonts_dir} to {duplicates_dir}")


# ----------------------------------------------------------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------------------------------------------------------

def parse_sweep(entries: Sequence[str], glyph_threshold: float) -> List[Tuple[float, float]]:
    """
    Parse ``--sweep`` entries, ``T`` or ``T:G``, into (mean threshold, glyph threshold) pairs.
    """
    result = []
    for entry in entries:
        t, _, g = entry.partition(":")
        result.append((float(t), float(g) if g else glyph_threshold))
    return result


def build_parser() -> argparse.ArgumentParser:
    """
    The command line interface.
    """
    p = argparse.ArgumentParser(description="Find (dry run, default) or remove near-duplicate fonts.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--fonts-dir", default="data/fonts", help="directory with the font files")
    p.add_argument("--annotations", default="data/fonts.csv", help="fonts.csv (font,file,supported_charset,family)")
    p.add_argument("--manifest", default="data/fonts_manifest.csv", help="manifest with licences, ignored if missing")
    p.add_argument("--report-dir", default="data/dedupe_report", help="where the dry run reports are written")
    p.add_argument("--duplicates-dir", default="data/fonts_duplicates", help="where --apply moves duplicate fonts to")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="largest mean absolute pixel difference (0..1) over all glyphs of two duplicate fonts")
    p.add_argument("--glyph-threshold", type=float, default=DEFAULT_GLYPH_THRESHOLD,
                   help="largest mean absolute pixel difference (0..1) of any single glyph of two duplicate fonts")
    p.add_argument("--sweep", nargs="*", default=[], metavar="T[:G]",
                   help="also print how many fonts would be removed for these thresholds")
    p.add_argument("--workers", type=int, default=3, help="processes for rendering and threads for comparing, max 4")
    p.add_argument("--no-cache", action="store_true", help="do not reuse or store fingerprints in the report dir")
    p.add_argument("--sheet-pairs", type=int, default=30, help="number of pairs in pairs.png")
    p.add_argument("--apply", action="store_true",
                   help="move the duplicate fonts to --duplicates-dir and rewrite the csv files (default: dry run)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Run the dry run or, with --apply, the removal.
    """
    args = build_parser().parse_args(argv)
    workers = max(1, min(4, args.workers))
    start = time.time()

    columns, rows, _ = read_csv(args.annotations)
    for needed in ("file", "font", "supported_charset"):
        if needed not in columns:
            print(f"ERROR: {args.annotations} has no column '{needed}'", file=sys.stderr)
            return 1
    manifest_rows: Dict[str, Dict[str, str]] = {}
    if args.manifest and os.path.exists(args.manifest):
        manifest_rows = {row["file"]: row for row in read_csv(args.manifest)[1]}
    else:
        print("No manifest found, every font counts as unknown licence.")
        args.manifest = None

    os.makedirs(args.report_dir, exist_ok=True)
    cache = None if args.no_cache else os.path.join(args.report_dir, "fingerprints.npz")
    print(f"Rendering fingerprints of {len(rows)} fonts with {workers} processes")
    prints, usable, skipped = compute_fingerprints(rows, args.fonts_dir, workers, cache)
    if skipped:
        with open(os.path.join(args.report_dir, "skipped_fonts.csv"), "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([("file", "reason"), *skipped])
        for name, reason in skipped:
            print(f"SKIPPED {name}: {reason}")
    if len(prints) < 2:
        print("Fewer than two usable fonts, nothing to compare.")
        return 1

    info = []
    for i in usable:
        row = rows[i]
        license_text = manifest_rows.get(row["file"], {}).get("license", "")
        info.append({"file": row["file"], "font": row["font"], "family": row.get("family", ""),
                     "license": license_text})
    files = [item["file"] for item in info]
    open_license = [is_open_license(item["license"]) for item in info]

    print(f"Comparing all {len(prints) * (len(prints) - 1) // 2} pairs of {len(prints)} fonts")
    mean_d, worst_d = pairwise_distances(prints, workers)

    plan = plan_removals(mean_d, worst_d, args.threshold, args.glyph_threshold, files, open_license)
    removed_indices = {i for _, removed in plan for i in removed}

    write_duplicates_csv(os.path.join(args.report_dir, "duplicates.csv"), plan, mean_d, worst_d, info)
    write_nearest_csv(os.path.join(args.report_dir, "nearest_neighbours.csv"), mean_d, worst_d, info, removed_indices)
    pairs = select_sheet_pairs(mean_d, args.threshold, args.sheet_pairs)
    write_pairs_png(os.path.join(args.report_dir, "pairs.png"), pairs, prints, mean_d, worst_d, info, args.threshold)

    print()
    print(f"fonts compared:          {len(prints)} ({len(skipped)} skipped)")
    print(f"mean threshold:          {args.threshold}")
    print(f"single glyph threshold:  {args.glyph_threshold}")
    print(f"duplicate groups:        {len(plan)}")
    print(f"fonts to remove:         {len(removed_indices)}")
    print(f"fonts left:              {len(prints) - len(removed_indices)}")
    for keeper, removed in plan[:5]:
        print(f"  group of {len(removed) + 1}: keep {files[keeper]!r}, remove {[files[i] for i in removed]}")
    for t, g in parse_sweep(args.sweep, args.glyph_threshold):
        other = plan_removals(mean_d, worst_d, t, g, files, open_license)
        print(f"  sweep threshold {t} / glyph {g}: {len(other)} groups, "
              f"{sum(len(r) for _, r in other)} fonts removed")
    print(f"reports written to {args.report_dir} ({time.time() - start:.0f}s)")

    if not args.apply:
        print("Dry run, nothing changed. Use --apply to move the duplicates away.")
        return 0
    if not removed_indices:
        print("Nothing to remove.")
        return 0
    apply_removals([files[i] for i in sorted(removed_indices)], args.fonts_dir, args.duplicates_dir,
                   args.annotations, args.manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
