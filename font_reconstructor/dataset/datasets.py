from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from .cache import cache_key, cached_array
from .fonts import FontSet
from .rendering import render_fingerprint, render_text

DEFAULT_CACHE_DIR = 'data/cache'


class RandomTextImageDataset(Dataset):
    """
    Images of random texts rendered with random fonts. Sample `idx` is fully determined by `random_seed`.

    Each sample is a dict with
        image:      uint8 array (height, width) of the rendered text
        target:     uint8 array (height, width, glyphs), the fingerprint of the font. Only if `return_target`.
        text:       the rendered text
        font:       name of the font
        font_index: position of the font in `fonts`

    With `group_by_font` the fonts are cycled in order, so every font gets the same number of samples.
    """

    def __init__(
        self,
        fonts: FontSet,
        random_seed: Optional[int] = None,
        total_samples: int = 10_000,
        text_length: Tuple[int, int] = (3, 8),
        text_image_dims: Tuple[int, int] = (128, 32),
        font_fingerprint_dims: Tuple[int, int] = (32, 32),
        group_by_font: bool = False,
        return_target: bool = True,
        cache_fingerprints: bool = True,
        cache_images: bool = True,
        cache_dir: str = DEFAULT_CACHE_DIR,
    ):
        if len(fonts) == 0:
            raise ValueError("The font set is empty, there is nothing to render.")

        self.fonts = fonts
        self.random_seed = random_seed
        self.total_samples = total_samples
        self.text_length = tuple(text_length)
        self.text_image_dims = tuple(text_image_dims)
        self.font_fingerprint_dims = tuple(font_fingerprint_dims)
        self.group_by_font = group_by_font
        self.return_target = return_target
        self.cache_dir = cache_dir

        if self.random_seed is None:
            self.random_seed = int(np.random.randint(0, 2**31 - 1))

        self._fingerprint_cache = None
        if return_target and cache_fingerprints:
            self._fingerprint_cache = self._prepare_fingerprint_cache()

        self._images_cache = None
        if cache_images:
            self._images_cache = self._prepare_images_cache()

    def _prepare_fingerprint_cache(self):
        key = cache_key(self.fonts.signature(), self.font_fingerprint_dims)
        width, height = self.font_fingerprint_dims

        def fill(array):
            for font_index in tqdm(range(len(self.fonts)), desc="Generating font fingerprints"):
                array[font_index] = self.generate_font_fingerprint(font_index)

        return cached_array(
            self.cache_dir, "fingerprints", key,
            shape=(len(self.fonts), height, width, self.fonts.num_glyphs), dtype=np.uint8, fill=fill,
        )

    def _prepare_images_cache(self):
        key = cache_key(
            self.fonts.signature(), self.random_seed, self.total_samples,
            self.text_length, self.text_image_dims, self.group_by_font,
        )
        width, height = self.text_image_dims

        def fill(array):
            for idx in tqdm(range(self.total_samples), desc="Generating text images"):
                font_index, text = self._sample_spec(idx)
                array[idx] = self.generate_text_image(font_index, text)

        return cached_array(
            self.cache_dir, "images", key,
            shape=(self.total_samples, height, width), dtype=np.uint8, fill=fill,
        )

    def generate_font_fingerprint(self, font_index):
        return render_fingerprint(
            self.fonts.ttf(font_index),
            self.fonts.glyphs(font_index),
            self.font_fingerprint_dims,
            self.fonts.num_glyphs,
        )

    def generate_text_image(self, font_index, text):
        return render_text(self.fonts.ttf(font_index), text, self.text_image_dims)

    def _sample_spec(self, idx):
        """
        font and text of sample `idx`, derived from the dataset's random seed and the index
        """
        if self.group_by_font:
            rand = np.random.RandomState(self.random_seed + idx // len(self.fonts))
            font_index = idx % len(self.fonts)
        else:
            rand = np.random.RandomState(self.random_seed + idx)
            font_index = rand.randint(0, len(self.fonts))

        text = _generate_rand_text(rand, self.text_length, self.fonts.charset(font_index))
        return font_index, text

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError

        font_index, text = self._sample_spec(idx)

        if self._images_cache is not None:
            image = self._images_cache[idx]
        else:
            image = self.generate_text_image(font_index, text)

        sample = {
            'image': image,
            'text': text,
            'font': self.fonts.names[font_index],
            'font_index': font_index,
        }

        if self.return_target:
            if self._fingerprint_cache is not None:
                sample['target'] = self._fingerprint_cache[font_index]
            else:
                sample['target'] = self.generate_font_fingerprint(font_index)

        return sample


class TransformedSubset(Dataset):
    """
    A subset of a sample-dict dataset with transforms applied to the image and the target.

    Different subsets of one dataset can carry different transforms, for example augmentations for training
    and none for validation.
    """

    def __init__(
        self,
        dataset: Dataset,
        indices: Optional[Sequence[int]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset) if self.indices is None else len(self.indices)

    def __getitem__(self, idx):
        if self.indices is not None:
            idx = int(self.indices[idx])
        elif idx < 0 or idx >= len(self.dataset):
            raise IndexError

        sample = dict(self.dataset[idx])
        if self.transform is not None:
            sample['image'] = self.transform(sample['image'])
        if self.target_transform is not None and 'target' in sample:
            sample['target'] = self.target_transform(sample['target'])
        return sample


def _generate_rand_text(rand, length_range, charset):
    characters = [*charset.replace("\0", "")]
    while True:
        length = rand.randint(*length_range)
        text = "".join(rand.choice(characters, length))
        if len(text.strip()) >= length_range[0]:
            return text
