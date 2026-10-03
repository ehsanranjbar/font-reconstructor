from typing import Literal, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def render_text(
    ttf: ImageFont.FreeTypeFont,
    text: str,
    dims: Tuple[int, int],
    align: Literal['left', 'center', 'right'] = 'center',
    fill: float = 1.0,
) -> np.ndarray:
    """
    Render white text on black, scaled to fit `dims` while keeping its aspect ratio.

    :param dims: (width, height) of the image
    :param fill: share of the image the text may take up. Below 1 a black border is left around the text,
        which later distortions of the image need as room.
    :return: uint8 array shaped (height, width). A black image if the text can not be rendered.
    """
    try:
        left, top, right, bottom = ttf.getbbox(text, anchor="lt")
        (width, height) = (right - left, bottom - top)

        img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), text, fill=255, anchor="lt", font=ttf)

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
