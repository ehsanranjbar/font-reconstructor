from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


@dataclass(frozen=True)
class Variant:
    """
    A way of drawing a font that turns it into another one: outlined, heavier or slanted.

    Real collections have few outline and slanted fonts. Drawing ordinary fonts this way gives many more of
    them, each with its own text images and its own fingerprint.

    :param outline: width of the outline as a fraction of the font size. The glyphs are drawn hollow, with a
        line of this width around them. 0 draws them filled.
    :param weight: extra stroke width as a fraction of the font size, which makes filled glyphs heavier
    :param slant: horizontal shift per unit of height. Positive values lean the top to the left, as the
        slanted styles of Persian fonts do.
    """
    outline: float = 0.0
    weight: float = 0.0
    slant: float = 0.0

    def signature(self):
        return [self.outline, self.weight, self.slant]


def _draw(ttf: ImageFont.FreeTypeFont, text: str, variant: Optional[Variant]) -> Image.Image:
    """
    the text in white on black, at the size of the font and cropped to its bounding box
    """
    stroke, hollow = 0, False
    if variant is not None:
        hollow = variant.outline > 0
        stroke = max(1, round(ttf.size * variant.outline)) if hollow else round(ttf.size * variant.weight)

    left, top, right, bottom = ttf.getbbox(text, anchor="lt", stroke_width=stroke)
    (width, height) = (right - left, bottom - top)

    # the bounding box is relative to the anchor and can start left of it or above it, for example for a
    # letter that reaches back under its neighbour. Drawing at its offset keeps all of the ink in the image.
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    # the stroke is drawn around the whole text first and the fill on top of it, so a black fill leaves
    # one outline around joined letters instead of an outline around each of them
    draw.text((-left, -top), text, fill=0 if hollow else 255, anchor="lt", font=ttf,
              stroke_width=stroke, stroke_fill=255)

    if variant is not None and variant.slant:
        # every row moves sideways in proportion to its distance from the top (or from the bottom, for a
        # negative slant), so the image gets wider by the shift of its last row
        shift = abs(variant.slant) * height
        offset = variant.slant * height if variant.slant < 0 else 0.0
        img = img.transform((width + int(np.ceil(shift)), height), Image.AFFINE,
                            (1, -variant.slant, offset, 0, 1, 0), Image.BICUBIC)
    return img


def render_text(
    ttf: ImageFont.FreeTypeFont,
    text: str,
    dims: Tuple[int, int],
    align: Literal['left', 'center', 'right'] = 'center',
    fill: float = 1.0,
    variant: Optional[Variant] = None,
) -> np.ndarray:
    """
    Render white text on black, scaled to fit `dims` while keeping its aspect ratio.

    :param dims: (width, height) of the image
    :param fill: share of the image the text may take up. Below 1 a black border is left around the text,
        which later distortions of the image need as room.
    :param variant: draw the font outlined, heavier or slanted
    :return: uint8 array shaped (height, width). A black image if the text can not be rendered.
    """
    try:
        img = _draw(ttf, text, variant)
        (width, height) = img.size

        inner = (max(1, int(dims[0] * fill)), max(1, int(dims[1] * fill)))
        scale = min(inner[0] / width, inner[1] / height)
        img.thumbnail(
            (int(width * scale), int(height * scale)), Image.LANCZOS)

        centering = (0.5, 0.5)
        if align == "left":
            centering = (0.0, 0.5)
        elif align == "right":
            centering = (1.0, 0.5)
        img = ImageOps.pad(img, inner, method=Image.LANCZOS,
                           centering=centering)
        if inner != tuple(dims):
            canvas = Image.new("L", tuple(dims), 0)
            canvas.paste(img, ((dims[0] - inner[0]) // 2, (dims[1] - inner[1]) // 2))
            img = canvas
    except Exception as e:
        font_name = "/".join(str(part) for part in ttf.getname())
        print(
            f"ERROR: Rendering text \"{text}\" with font \"{font_name}\" failed with exception \"{e}\"")
        img = Image.new("L", tuple(dims), 0)

    return np.array(img)


def render_fingerprint(
    ttf: ImageFont.FreeTypeFont,
    glyphs: str,
    dims: Tuple[int, int],
    num_glyphs: int,
    variant: Optional[Variant] = None,
) -> np.ndarray:
    """
    Render each glyph to its own channel.

    :param glyphs: characters of the font. Null characters leave their channel empty.
    :param dims: (width, height) of each glyph image
    :param num_glyphs: number of channels, at least len(glyphs)
    :param variant: draw the font outlined, heavier or slanted
    :return: uint8 array shaped (height, width, num_glyphs)
    """
    fingerprint = np.zeros((dims[1], dims[0], num_glyphs), dtype=np.uint8)
    for i, char in enumerate(glyphs):
        # Ignore null characters
        if char == "\0":
            continue

        fingerprint[:, :, i] = render_text(ttf, char, dims, variant=variant)

    return fingerprint
