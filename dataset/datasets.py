import hashlib
import math
import os
from typing import Literal, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import Dataset
from tqdm import tqdm

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
        group_by_font: bool = False,
        cache_fingerprints: bool = True,
        cache_images: bool = True,
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
        self._group_by_font = group_by_font
        self._cache_fingerprints = cache_fingerprints
        self._cache_images = cache_images

        if self.random_seed is None:
            self.random_seed = np.random.randint()

        self._font_fingerprint_length = self._calc_font_fingerprint_length()
        self._load_ttfs()

        if self._cache_fingerprints:
            self._prepare_font_fingerprint_cache()

        if self._cache_images:
            self._prepare_images_cache()

    def _calc_font_fingerprint_length(self):
        return max([len(charset.replace(" ", "")) for charset in self._fonts["supported_charset"]])

    def _load_ttfs(self):
        ttfs = []
        ignored_fonts = 0
        for index, row in tqdm(self._fonts.iterrows(), total=len(self._fonts), desc="Loading & validating fonts"):
            file = row["file"]
            supported_charset = row["supported_charset"].replace(" ", "")

            try:
                font_path = os.path.join(self.fonts_dir, file)
                ttf = ImageFont.truetype(font_path, self.font_size, encoding="unic")
            except Exception as e:
                tqdm.write(f"ERROR: Failed to open {font_path} as TrueType font with exception \"{e}\"")

                self._fonts.drop(index, inplace=True)
                continue

            unsupported_chars = _check_font_charset_support(ttf, supported_charset)
            if len(unsupported_chars) > 0:
                tqdm.write(
                    f"WARN: Font {file} ignored from fonts list because it does not support {unsupported_chars} from charset"
                )

                self._fonts.drop(index, inplace=True)
                ignored_fonts += 1
                continue

            ttfs.append(ttf)

        tqdm.write(f"Loaded {len(ttfs)} fonts, ignored {ignored_fonts}")
        self._ttfs = ttfs

    def _prepare_font_fingerprint_cache(self):
        # Get shorted md5 hash of fonts dataframe and font size
        cache_hash = hashlib.md5(pd.util.hash_pandas_object(self._fonts, index=True).to_numpy().tobytes())
        cache_hash.update(str(self.font_size).encode("utf-8"))
        cache_hash = cache_hash.hexdigest()[:8]

        # Retrieve fingerprints numpy array from cache if available
        cache_path = os.path.join(
            _PYTHON_CACHE_DIR, f"ff_{cache_hash}.npy")
        if os.path.exists(cache_path):
            print(
                f"Font fingerprints cache found at {cache_path}, loading from cache")

            self._fingerprint_cache = np.load(cache_path, allow_pickle=True)
            return

        self._fingerprint_cache = []
        for index, row in tqdm(self._fonts.iterrows(), total=len(self._fonts), desc="Generating font fingerprints"):
            tqdm.write(row["font"])

            self._fingerprint_cache.append(self.generate_font_fingerprint(index))

        # Save fingerprints numpy array to cache
        print(f"Saving font fingerprints cache to {cache_path}")
        os.makedirs(_PYTHON_CACHE_DIR, exist_ok=True)
        np.save(cache_path, self._fingerprint_cache)

    def generate_font_fingerprint(self, font_index):
        fingerprint = np.zeros(
            (*self.font_fingerprint_dims, self._font_fingerprint_length), dtype=np.uint8
        )
        supported_charset = self._fonts.iloc[font_index]["supported_charset"].replace(" ", "")
        for i, char in enumerate(supported_charset):
            # Ignore null characters
            if char == "\0":
                continue

            fingerprint[:, :, i] = self._generate_text_image(
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
            print(
                f"ERROR: Rendering text \"{text}\" with font \"{font}\" failed with exception \"{e}\"")
            img = Image.new("L", dims, 0)

        return np.array(img)

    def _prepare_images_cache(self):
        # Get shorted md5 hash of self properties
        cache_hash = hashlib.md5()
        cache_hash.update(pd.util.hash_pandas_object(self._fonts, index=True).to_numpy().tobytes())
        cache_hash.update(str(self.random_seed).encode("utf-8"))
        cache_hash.update(str(self.total_samples).encode("utf-8"))
        cache_hash.update(str(self.font_size).encode("utf-8"))
        cache_hash.update(str(self.text_length).encode("utf-8"))
        cache_hash.update(str(self.text_image_dims).encode("utf-8"))
        cache_hash.update(str(self.font_fingerprint_dims).encode("utf-8"))
        cache_hash.update(str(self._group_by_font).encode("utf-8"))
        cache_hash = cache_hash.hexdigest()[:8]

        # Retrieve images numpy array from cache if available
        cache_path = os.path.join(
            _PYTHON_CACHE_DIR, f"im_{cache_hash}.npy")
        if os.path.exists(cache_path):
            print(
                f"Text images cache found at {cache_path}, loading from cache")

            self._images_cache = np.load(cache_path, allow_pickle=True)
            return

        self._cache_images = False
        transform = self.transform
        target_transform = self.target_transform
        self.transform = None
        self.target_transform = None
        self._images_cache = []
        for (img, _, _, _) in tqdm(self, total=self.total_samples, desc="Generating text images"):
            self._images_cache.append(img)
        self._cache_images = True
        self.transform = transform
        self.target_transform = target_transform

        # Save images numpy array to cache
        print(f"Saving text images cache to {cache_path}")
        os.makedirs(_PYTHON_CACHE_DIR, exist_ok=True)
        np.save(cache_path, self._images_cache)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError

        # Generate a random generator from the dataset's random seed and index
        if self._group_by_font:
            rand = np.random.RandomState(self.random_seed + math.floor(idx / len(self._fonts)))
            font_index = idx % len(self._fonts)
        else:
            rand = np.random.RandomState(self.random_seed + idx)
            font_index = rand.randint(0, len(self._fonts))

        # Generate a random text using the supported charset of font
        supported_charset = self._fonts.iloc[font_index]["supported_charset"]
        text = _generate_rand_text(rand, self.text_length, supported_charset)
        X = self._get_text_image(text, font_index, idx=idx)

        y = self._get_font_fingerprint(font_index)

        font = self._fonts.iloc[font_index]["font"]

        if self.transform is not None:
            X = self.transform(X)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return X, y, text, font

    def _get_text_image(self, text, font_index, idx=None):
        if self._cache_images and idx is not None:
            return self._images_cache[idx]
        else:
            return self._generate_text_image(font_index, text, self.text_image_dims)

    def _get_font_fingerprint(self, font_index):
        if self._cache_fingerprints:
            return self._fingerprint_cache[font_index]
        else:
            return self.generate_font_fingerprint(font_index)


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
