import os
from typing import List

import pandas as pd
from PIL import ImageFont
from tqdm import tqdm


class FontSet:
    """
    The fonts listed in an annotation csv file, validated against their declared charset.

    The annotation file needs the columns `font` (display name), `file` (path relative to `fonts_dir`) and
    `supported_charset`. Spaces in the charset are ignored for the fingerprint. A null character keeps a glyph
    slot empty, which keeps glyph positions aligned between fonts that support different characters.

    Fonts that fail to open or miss a glyph of their charset are dropped. Fonts are addressed by their position
    in the remaining list.
    """

    def __init__(self, fonts_dir: str, annotation_file: str, font_size: int = 32):
        self.fonts_dir = fonts_dir
        self.font_size = font_size

        annotations = pd.read_csv(annotation_file)
        # the fingerprint length covers every annotated font, so it does not depend on which fonts get dropped
        self.num_glyphs = max(len(_glyphs(charset)) for charset in annotations["supported_charset"])

        self.names: List[str] = []
        self.files: List[str] = []
        self.charsets: List[str] = []
        self._ttfs = {}

        ignored_fonts = 0
        for _, row in tqdm(annotations.iterrows(), total=len(annotations), desc="Loading & validating fonts"):
            file = row["file"]
            charset = row["supported_charset"]

            font_path = os.path.join(self.fonts_dir, file)
            try:
                ttf = ImageFont.truetype(font_path, self.font_size, encoding="unic")
            except Exception as e:
                tqdm.write(f"ERROR: Failed to open {font_path} as TrueType font with exception \"{e}\"")
                ignored_fonts += 1
                continue

            unsupported_chars = unsupported_glyphs(ttf, _glyphs(charset))
            if len(unsupported_chars) > 0:
                tqdm.write(
                    f"WARN: Font {file} ignored from fonts list because it does not support {unsupported_chars} from charset"
                )
                ignored_fonts += 1
                continue

            self._ttfs[len(self.names)] = ttf
            self.names.append(row["font"])
            self.files.append(file)
            self.charsets.append(charset)

        tqdm.write(f"Loaded {len(self.names)} fonts, ignored {ignored_fonts}")

    def __len__(self):
        return len(self.names)

    def charset(self, index: int) -> str:
        """
        characters random texts are drawn from, as written in the annotation file
        """
        return self.charsets[index]

    def glyphs(self, index: int) -> str:
        """
        characters of the font fingerprint, one per channel
        """
        return _glyphs(self.charsets[index])

    def ttf(self, index: int) -> ImageFont.FreeTypeFont:
        # fonts are opened again on first use in each worker process, see __getstate__
        if index not in self._ttfs:
            font_path = os.path.join(self.fonts_dir, self.files[index])
            self._ttfs[index] = ImageFont.truetype(font_path, self.font_size, encoding="unic")
        return self._ttfs[index]

    def signature(self):
        """
        everything that determines what this font set renders, used to key caches
        """
        return [self.font_size, self.num_glyphs, self.names, self.files, self.charsets]

    def __getstate__(self):
        # opened font handles are not sent to data loader workers
        state = self.__dict__.copy()
        state['_ttfs'] = {}
        return state


def unsupported_glyphs(ttf: ImageFont.FreeTypeFont, glyphs: str) -> List[str]:
    # null characters mark glyph slots that are left empty on purpose
    return [c for c in glyphs if c != "\0" and not _ttf_support_glyph(ttf, c)]


def _ttf_support_glyph(ttf: ImageFont.FreeTypeFont, glyph: str) -> bool:
    left, top, right, bottom = ttf.getbbox(glyph, anchor="lt")
    (width, height) = (right - left, bottom - top)
    return width > 0 and height > 0


def _glyphs(charset: str) -> str:
    return charset.replace(" ", "")
