from typing import Literal, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def render_text(
    ttf: ImageFont.FreeTypeFont,
    text: str,
    dims: Tuple[int, int],
    align: Literal['left', 'center', 'right'] = 'center',
) -> np.ndarray:
    """
    Render white text on black, scaled to fit `dims` while keeping its aspect ratio.

    :param dims: (width, height) of the image
    :return: uint8 array shaped (height, width). A black image if the text can not be rendered.
    """
    try:
        left, top, right, bottom = ttf.getbbox(text, anchor="lt")
        (width, height) = (right - left, bottom - top)

        img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), text, fill=255, anchor="lt", font=ttf)

        scale = min(dims[0] / width, dims[1] / height)
        img.thumbnail(
            (int(width * scale), int(height * scale)), Image.LANCZOS)

        centering = (0.5, 0.5)
        if align == "left":
            centering = (0.0, 0.5)
        elif align == "right":
            centering = (1.0, 0.5)
        img = ImageOps.pad(img, tuple(dims), method=Image.LANCZOS,
                           centering=centering)
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
) -> np.ndarray:
    """
    Render each glyph to its own channel.

    :param glyphs: characters of the font. Null characters leave their channel empty.
    :param dims: (width, height) of each glyph image
    :param num_glyphs: number of channels, at least len(glyphs)
    :return: uint8 array shaped (height, width, num_glyphs)
    """
    fingerprint = np.zeros((dims[1], dims[0], num_glyphs), dtype=np.uint8)
    for i, char in enumerate(glyphs):
        # Ignore null characters
        if char == "\0":
            continue

        fingerprint[:, :, i] = render_text(ttf, char, dims)

    return fingerprint
