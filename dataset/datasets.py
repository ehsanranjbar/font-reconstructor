import hashlib
import logging
import os
from typing import Literal, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import Dataset
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

_logger = logging.getLogger(__name__)

_PYTHON_CACHE_DIR = '__pycache__'


class RandomTextImageDataset(Dataset):
    def __init__(
        self,
        fonts_dir: str,
        annotation_file: str,
        random_seed: int = None,
        total_samples: int = 10_000,
        font_size: int = 32,
        text_length: Tuple[int, int] = (3, 8),
        text_image_dims: Tuple[int, int] = (128, 32),
        font_fingerprint_dims: Tuple[int, int] = (32, 32),
        transform=None,
        target_transform=None,
        use_cache: bool = True,
    ):
        self.fonts_dir = fonts_dir
        self._fonts = pd.read_csv(annotation_file)
        self.random_seed = random_seed
        self.total_samples = total_samples
        self.font_size = font_size
        self.text_length = text_length
        self.text_image_dims = text_image_dims
        self.font_fingerprint_dims = font_fingerprint_dims
        self.transform = transform
        self.target_transform = target_transform
        self._use_cache = use_cache

        if self.random_seed is None:
            self.random_seed = np.random.randint()

        self._font_fingerprint_length = self._calc_font_fingerprint_length()
        self._load_ttfs()
        self._prepare_font_fingerprint_cache()

    def _calc_font_fingerprint_length(self):
        return max([len(charset.replace(" ", "")) for charset in self._fonts["supported_charset"]])

    def _load_ttfs(self):
        ttfs = []
        ignored_fonts = 0
        with logging_redirect_tqdm():
            for index, row in tqdm(self._fonts.iterrows(), desc="Loading & validating fonts"):
                file = row["file"]
                supported_charset = row["supported_charset"].replace(" ", "")

                try:
                    font_path = os.path.join(self.fonts_dir, file)
                    ttf = ImageFont.truetype(font_path, self.font_size, encoding="unic")
                except Exception as e:
                    _logger.error(f"Failed to open {font_path} as TrueType font with exception \"{e}\"")

                    self._fonts.drop(index, inplace=True)
                    continue

                unsupported_chars = _check_font_charset_support(ttf, supported_charset)
                if len(unsupported_chars) > 0:
                    _logger.warning(
                        f"Font {file} ignored from fonts list because it does not support {unsupported_chars} from charset"
                    )

                    self._fonts.drop(index, inplace=True)
                    ignored_fonts += 1
                    continue

                ttfs.append(ttf)

            _logger.info(
                f"Loaded {len(ttfs)} fonts, ignored {ignored_fonts}")
            self._ttfs = ttfs

    def _prepare_font_fingerprint_cache(self):
        if self._use_cache:
            # Get shorted md5 hash of fonts dataframe and font size
            cache_hash = hashlib.md5(pd.util.hash_pandas_object(self._fonts, index=True).to_numpy().tobytes())
            cache_hash.update(str(self.font_size).encode("utf-8"))
            cache_hash = cache_hash.hexdigest()[:8]

            # Retrieve fingerprints numpy array from cache if available
            cache_path = os.path.join(
                _PYTHON_CACHE_DIR, f"ff_{cache_hash}.npy")
            if os.path.exists(cache_path):
                _logger.info(
                    f"Font fingerprints cache found at {cache_path}, loading from cache")

                self._fingerprint_cache = np.load(cache_path, allow_pickle=True)
                return

        self._fingerprint_cache = []
        with logging_redirect_tqdm():
            for index, row in tqdm(self._fonts.iterrows(), desc="Generating font fingerprints"):
                _logger.info(row["font"])

                self._fingerprint_cache.append(self.generate_font_fingerprint(index))

        if self._use_cache:
            # Save fingerprints numpy array to cache
            _logger.info(f"Saving font fingerprints cache to {cache_path}")
            os.makedirs(_PYTHON_CACHE_DIR, exist_ok=True)
            np.save(cache_path, self._fingerprint_cache)

    def generate_font_fingerprint(self, font_index):
        fingerprint = np.zeros(
            (self._font_fingerprint_length, *self.font_fingerprint_dims), dtype=np.uint8
        )
        supported_charset = self._fonts.iloc[font_index]["supported_charset"].replace(" ", "")
        for i, char in enumerate(supported_charset):
            # Ignore null characters
            if char == "\0":
                continue

            fingerprint[i, ] = self._generate_text_image(
                font_index,
                char,
                self.font_fingerprint_dims,
            )

        return fingerprint

    def _generate_text_image(
        self,
        font_index,
        text: str,
        dims: Tuple[int, int],
        align: Literal['left', 'center', 'right'] = 'center',
    ):
        font = self._fonts.iloc[font_index]["font"]
        ttfont = self._ttfs[font_index]

        try:
            left, top, right, bottom = ttfont.getbbox(text, anchor="lt")
            (width, height) = (right - left, bottom - top)

            # Draw the text in center of img
            img = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), text, fill=255, anchor="lt", font=ttfont)

            scale = min(dims[0] / width, dims[1] / height)
            img.thumbnail(
                (int(width * scale), int(height * scale)), Image.LANCZOS)

            centering = (0.5, 0.5)
            if align == "left":
                centering = (0.0, 0.5)
            elif align == "right":
                centering = (1.0, 0.5)
            img = ImageOps.pad(img, dims, method=Image.LANCZOS,
                               centering=centering)
        except Exception as e:
            _logger.error(
                f"Rendering text \"{text}\" with font \"{font}\" failed with exception \"{e}\"")
            img = Image.new("L", dims, 0)

        return np.array(img)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError

        # Generate a random generator from the dataset's random seed and index
        rand = np.random.RandomState(self.random_seed + idx)

        # Select a random index for font
        font_index = rand.randint(0, len(self._fonts))

        # Generate a random text using the supported charset of font
        supported_charset = self._fonts.iloc[font_index]["supported_charset"]
        text = _generate_rand_text(rand, self.text_length, supported_charset)

        text_image = self._generate_text_image(font_index, text, self.text_image_dims)

        return text_image, self._fingerprint_cache[font_index]


def _check_font_charset_support(ttf, supported_charset):
    unsupported_chars = []
    for c in supported_charset:
        if not _ttf_support_glyph(ttf, c):
            unsupported_chars.append(c)

    return unsupported_chars


def _ttf_support_glyph(font: ImageFont.FreeTypeFont, glyph: str):
    left, top, right, bottom = font.getbbox(glyph, anchor="lt")
    (width, height) = (right - left, bottom - top)
    return width > 0 and height > 0


def _generate_rand_text(rand, length_range, charset):
    length = rand.randint(*length_range)
    text = "".join(rand.choice([*charset.replace("\0", "")], length))
    if len(text.strip()) < length_range[0]:
        return _generate_rand_text(rand, length_range, charset)

    return text
