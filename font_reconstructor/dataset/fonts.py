import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import ImageFont, features
from tqdm import tqdm

from .rendering import Variant

# How a synthetic font of each style is drawn, and which real fonts it can be made from. The widths are
# fractions of the font size.
SYNTHETIC_STYLES = {
    'outline': (Variant(outline=0.03), ('regular', 'bold', 'light')),
    'italic': (Variant(slant=0.2), ('regular',)),
    'bold italic': (Variant(slant=0.2), ('bold',)),
    'bold': (Variant(weight=0.03), ('regular',)),
}

_LAYOUT_ENGINES = {
    'basic': ImageFont.Layout.BASIC,
    'raqm': ImageFont.Layout.RAQM,
}

# weight and style words that some fonts append to their family name
_STYLE_WORDS = re.compile(
    r'[\s\-_]*(thin|hairline|extra[\s\-]?light|ultra[\s\-]?light|light|regular|normal|book|medium|'
    r'semi[\s\-]?bold|demi[\s\-]?bold|extra[\s\-]?bold|ultra[\s\-]?bold|bold|black|heavy|italic|oblique)$',
    re.IGNORECASE,
)


_WORDS = re.compile(r'[A-Z]+(?![a-z])|[A-Z]?[a-z]+')
_LIGHT_WORDS = {'thin', 'hairline', 'light'}
_BOLD_WORDS = {'bold', 'black', 'heavy', 'fat'}
_ITALIC_WORDS = {'italic', 'oblique'}
_OUTLINE_WORDS = {'outline', 'border', 'hollow', 'inline'}


def derive_style(*names: str) -> str:
    """
    Name the style of a font from the words of its names, for example its display name and the style name
    stored in the font file.

    Decorated fonts are 'outline' or 'shadow'. All others are 'light', 'regular' or 'bold' by their weight,
    followed by 'italic' if they are slanted. A regular italic is just 'italic'. Neighbouring weights are
    grouped: thin and extra light count as light, semi bold to black as bold, medium as regular.
    """
    words = {word.lower() for name in names for word in _WORDS.findall(str(name))}
    if words & _OUTLINE_WORDS:
        return 'outline'
    if 'shadow' in words:
        return 'shadow'

    weight = 'bold' if words & _BOLD_WORDS else 'light' if words & _LIGHT_WORDS else 'regular'
    if words & _ITALIC_WORDS:
        return 'italic' if weight == 'regular' else f'{weight} italic'
    return weight


def resolve_layout_engine(layout_engine: str = 'auto') -> str:
    """
    Pick the text layout engine of Pillow.

    'raqm' shapes text: letters of scripts such as Arabic and Persian are joined and written right to left.
    'basic' draws every character in its isolated form from left to right, which is only correct for scripts
    that need no shaping. 'auto' uses Raqm if this Pillow installation has it and basic layout otherwise.

    :return: 'raqm' or 'basic'
    """
    has_raqm = bool(features.check('raqm'))
    if layout_engine == 'auto':
        return 'raqm' if has_raqm else 'basic'
    if layout_engine not in _LAYOUT_ENGINES:
        raise ValueError(f"Unknown layout engine '{layout_engine}'. Valid options are 'auto', 'raqm' and 'basic'.")
    if layout_engine == 'raqm' and not has_raqm:
        raise RuntimeError(
            "The 'raqm' layout engine is requested, but this Pillow installation can not use it. Without it, "
            "Arabic and Persian letters are drawn unjoined and from left to right. Pillow needs the FriBiDi "
            "library at runtime: install it (macOS: 'brew install fribidi', Debian/Ubuntu: 'apt install "
            "libfribidi0') and make sure it is on the library search path. Check with "
            "'python -c \"from PIL import features; print(features.check(\\\"raqm\\\"))\"'. "
            "Set layout_engine to 'basic' to render without shaping."
        )
    return layout_engine


class FontSet:
    """
    The fonts listed in an annotation csv file, validated against their declared charset.

    The annotation file needs the columns `font` (display name), `file` (path relative to `fonts_dir`) and
    `supported_charset`. Spaces in the charset are ignored for the fingerprint. A null character keeps a glyph
    slot empty, which keeps glyph positions aligned between fonts that support different characters.

    An optional `family` column groups the weights and styles of one typeface. Fonts of one family are never
    split between training and validation. Without the column the family is read from the font file.

    An optional `style` column labels the style of each font, which the style head of the model learns to
    predict. Any set of labels works, for example calligraphic styles. Without the column the style is derived
    from the names of the font, see `derive_style`.

    Fonts that fail to open or miss a glyph of their charset are dropped. Fonts are addressed by their position
    in the remaining list.

    :param synthetic_variants: dict of style to the number of extra fonts to make of that style, for example
        {"outline": 200, "italic": 120}. Each one is a real font drawn outlined, slanted or heavier, see
        SYNTHETIC_STYLES. It is a font of its own, with its own text images and fingerprint, in the family of
        the font it is made from. This fills up the styles that a collection has few fonts of.
    :param variant_seed: seed of the choice of the fonts that the variants are made from
    """

    def __init__(self, fonts_dir: str, annotation_file: str, font_size: int = 32, layout_engine: str = 'auto',
                 synthetic_variants: Optional[Dict[str, int]] = None, variant_seed: int = 0):
        self.fonts_dir = fonts_dir
        self.font_size = font_size
        self.layout_engine = resolve_layout_engine(layout_engine)

        annotations = pd.read_csv(annotation_file)
        has_family = "family" in annotations.columns
        has_style = "style" in annotations.columns
        # the fingerprint length covers every annotated font, so it does not depend on which fonts get dropped
        self.num_glyphs = max(len(_glyphs(charset)) for charset in annotations["supported_charset"])

        self.names: List[str] = []
        self.files: List[str] = []
        self.charsets: List[str] = []
        self.families: List[str] = []
        self.styles: List[str] = []
        self._ttfs = {}

        ignored_fonts = 0
        for _, row in tqdm(annotations.iterrows(), total=len(annotations), desc="Loading & validating fonts"):
            file = row["file"]
            charset = row["supported_charset"]

            font_path = os.path.join(self.fonts_dir, file)
            try:
                ttf = self._open(font_path)
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

            family = row["family"] if has_family and isinstance(row["family"], str) else None
            self._ttfs[len(self.names)] = ttf
            self.names.append(row["font"])
            self.files.append(file)
            self.charsets.append(charset)
            self.families.append(family or family_name(ttf) or str(row["font"]))
            style = row["style"] if has_style and isinstance(row["style"], str) else None
            self.styles.append(style or derive_style(row["font"], ttf.getname()[1]))

        # how each font is drawn: None for a real font, a Variant for a synthetic one
        self.variants: List[Optional[Variant]] = [None] * len(self.names)
        real_fonts = len(self.names)
        self._add_synthetic_variants(synthetic_variants or {}, variant_seed)

        # the classes of the style head, in a fixed order
        self.style_names: List[str] = sorted(set(self.styles))
        self.style_indices: List[int] = [self.style_names.index(style) for style in self.styles]

        synthetic = len(self.names) - real_fonts
        tqdm.write(f"Loaded {real_fonts} fonts of {len(set(self.families))} families and "
                   f"{len(self.style_names)} styles, ignored {ignored_fonts}")
        if synthetic:
            tqdm.write(f"Made {synthetic} synthetic variants of them, {len(self.names)} fonts in all")

    def _add_synthetic_variants(self, counts: Dict[str, int], seed: int):
        """
        append fonts that are real fonts drawn as another style
        """
        rng = np.random.default_rng(seed)
        real_styles = list(self.styles)
        for style, count in counts.items():
            if style not in SYNTHETIC_STYLES:
                raise ValueError(f"No synthetic variant makes the style '{style}'. "
                                 f"Valid options are {sorted(SYNTHETIC_STYLES)}.")
            variant, source_styles = SYNTHETIC_STYLES[style]
            sources = [index for index, source in enumerate(real_styles) if source in source_styles]
            for index in rng.permutation(sources)[:count]:
                self.names.append(f"{self.names[index]} ({style}, synthetic)")
                self.files.append(self.files[index])
                self.charsets.append(self.charsets[index])
                self.families.append(self.families[index])
                self.styles.append(style)
                self.variants.append(variant)

    def is_synthetic(self, index: int) -> bool:
        return self.variants[index] is not None

    def _open(self, font_path: str) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(
            font_path, self.font_size, encoding="unic", layout_engine=_LAYOUT_ENGINES[self.layout_engine])

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
            self._ttfs[index] = self._open(os.path.join(self.fonts_dir, self.files[index]))
        return self._ttfs[index]

    def signature(self):
        """
        everything that determines what this font set renders, used to key caches
        """
        variants = [None if variant is None else variant.signature() for variant in self.variants]
        return [self.font_size, self.num_glyphs, self.layout_engine, self.names, self.files, self.charsets, variants]

    def __getstate__(self):
        # opened font handles are not sent to data loader workers
        state = self.__dict__.copy()
        state['_ttfs'] = {}
        return state


def family_name(ttf: ImageFont.FreeTypeFont) -> str:
    """
    the family name stored in the font file, without weight or style words appended to it
    """
    name = (ttf.getname()[0] or '').strip()
    while True:
        stripped = _STYLE_WORDS.sub('', name).strip()
        if stripped == name or not stripped:
            return name
        name = stripped


def unsupported_glyphs(ttf: ImageFont.FreeTypeFont, glyphs: str) -> List[str]:
    # null characters mark glyph slots that are left empty on purpose
    return [c for c in glyphs if c != "\0" and not _ttf_support_glyph(ttf, c)]


def _ttf_support_glyph(ttf: ImageFont.FreeTypeFont, glyph: str) -> bool:
    left, top, right, bottom = ttf.getbbox(glyph, anchor="lt")
    (width, height) = (right - left, bottom - top)
    return width > 0 and height > 0


def _glyphs(charset: str) -> str:
    return charset.replace(" ", "")
