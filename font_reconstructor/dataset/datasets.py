from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from .cache import cache_key, cached_array
from .corpus import TextCorpus
from .fonts import FontSet
from .rendering import render_fingerprint, render_text

DEFAULT_CACHE_DIR = 'data/cache'
# share of the rendered image that the text takes up. The border leaves room for rotating and shifting the text.
RENDER_FILL = 0.85
_DIGITS = '۰۱۲۳۴۵۶۷۸۹0123456789'


class RandomTextImageDataset(Dataset):
    """
    Images of random texts rendered with random fonts. Sample `idx` is fully determined by `random_seed`.

    Each sample is a dict with
        image:      uint8 array of the text rendered white on black. It is `render_scale` times the size of
                    `text_image_dims` with a border around the text, see TextImageTransform for the next step.
        target:     uint8 array (height, width, glyphs), the fingerprint of the font. Only if `return_target`.
        text:       the rendered text
        font:       name of the font
        font_index: position of the font in `fonts`
        style_index: position of the font's style in `fonts.style_names`

    With `group_by_font` the fonts are cycled in order, so every font gets the same number of samples.

    :param font_indices: positions in `fonts` of the fonts to draw from, all fonts if None. A subset keeps
        the positions of the full font set, so `font_index` means the same font in every dataset.
    :param corpus: real-world text to draw the texts from. Random characters of the font's charset are used
        without it, and for fonts the corpus has no fitting text for.
    :param number_ratio: share of the texts that get a number of one to four digits added, if the charset of
        the font has digits. A corpus of prose has few numbers, real text images have many.
    :param render_scale: the texts are rendered this many times larger than `text_image_dims`. Distortions
        that are applied before the image is scaled down are finer than a pixel of the model input then.
    """

    def __init__(
        self,
        fonts: FontSet,
        font_indices: Optional[Sequence[int]] = None,
        corpus: Optional[TextCorpus] = None,
        random_seed: Optional[int] = None,
        total_samples: int = 10_000,
        text_length: Tuple[int, int] = (3, 8),
        text_image_dims: Tuple[int, int] = (128, 32),
        font_fingerprint_dims: Tuple[int, int] = (32, 32),
        number_ratio: float = 0.0,
        render_scale: int = 2,
        group_by_font: bool = False,
        return_target: bool = True,
        cache_fingerprints: bool = True,
        cache_images: bool = True,
        cache_dir: str = DEFAULT_CACHE_DIR,
    ):
        if len(fonts) == 0:
            raise ValueError("The font set is empty, there is nothing to render.")

        self.fonts = fonts
        self._all_fonts = font_indices is None
        self.font_indices = list(range(len(fonts))) if font_indices is None else [int(i) for i in font_indices]
        if len(self.font_indices) == 0:
            raise ValueError("font_indices is empty, there is nothing to render.")
        self.corpus = corpus
        self.random_seed = random_seed
        self.total_samples = total_samples
        self.text_length = tuple(text_length)
        self.text_image_dims = tuple(text_image_dims)
        self.font_fingerprint_dims = tuple(font_fingerprint_dims)
        self.number_ratio = number_ratio
        self.render_scale = int(render_scale)
        self.render_dims = (self.text_image_dims[0] * self.render_scale, self.text_image_dims[1] * self.render_scale)
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
            None if self._all_fonts else self.font_indices,
            None if self.corpus is None else self.corpus.signature(),
            self.number_ratio, self.render_scale, RENDER_FILL,
        )
        width, height = self.render_dims

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

    def font_fingerprint(self, font_index):
        """
        the fingerprint of a font as a uint8 array (height, width, glyphs), from the cache if there is one
        """
        if self._fingerprint_cache is not None:
            return self._fingerprint_cache[font_index]
        return self.generate_font_fingerprint(font_index)

    def generate_text_image(self, font_index, text):
        return render_text(self.fonts.ttf(font_index), text, self.render_dims, fill=RENDER_FILL)

    def _sample_spec(self, idx):
        """
        font and text of sample `idx`, derived from the dataset's random seed and the index
        """
        n_fonts = len(self.font_indices)
        if self.group_by_font:
            rand = np.random.RandomState(self.random_seed + idx // n_fonts)
            font_index = self.font_indices[idx % n_fonts]
        else:
            rand = np.random.RandomState(self.random_seed + idx)
            font_index = self.font_indices[rand.randint(0, n_fonts)]

        charset = self.fonts.charset(font_index)
        text = None
        if self.corpus is not None:
            text = self.corpus.sample(rand, self.text_length, charset)
        if text is None:
            text = _generate_rand_text(rand, self.text_length, charset)
        if self.number_ratio and rand.random_sample() < self.number_ratio:
            text = _add_number(rand, text, self.text_length, charset)
        return font_index, text

    def sample_indices_by_font(self):
        """
        The sample indices of each font, in the order of `font_indices`.

        Only available with `group_by_font`, where the font of a sample follows from its index.
        """
        if not self.group_by_font:
            raise ValueError("sample_indices_by_font needs a dataset with group_by_font=True.")
        n_fonts = len(self.font_indices)
        return [np.arange(position, self.total_samples, n_fonts) for position in range(n_fonts)]

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
            'style_index': self.fonts.style_indices[font_index],
        }

        if self.return_target:
            sample['target'] = self.font_fingerprint(font_index)

        return sample


class TransformedSubset(Dataset):
    """
    A subset of a sample-dict dataset with transforms applied to the image and the target.

    Different subsets of one dataset can carry different transforms, for example augmentations for training
    and none for validation.

    :param deterministic: call the image transform with `seed=<sample index>`, so that a random transform
        gives every sample the same result each time it is read. The transform has to accept `seed`.
    """

    def __init__(
        self,
        dataset: Dataset,
        indices: Optional[Sequence[int]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        deterministic: bool = False,
    ):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.target_transform = target_transform
        self.deterministic = deterministic

    def __len__(self):
        return len(self.dataset) if self.indices is None else len(self.indices)

    def __getitem__(self, idx):
        if self.indices is not None:
            idx = int(self.indices[idx])
        elif idx < 0 or idx >= len(self.dataset):
            raise IndexError

        sample = dict(self.dataset[idx])
        if self.transform is not None:
            if self.deterministic:
                sample['image'] = self.transform(sample['image'], seed=idx)
            else:
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


def _add_number(rand, text, length_range, charset):
    """
    Put a number of one to four digits before, inside or after the text, keeping the text length in range.

    Words are dropped from the end to make room. The text is returned unchanged if the charset has no digits.
    """
    digits = [char for char in charset if char in _DIGITS]
    if not digits:
        return text

    min_length, max_length = length_range
    number = "".join(rand.choice(digits, rand.randint(1, 5)))
    words = text.split()
    position = rand.randint(0, len(words) + 1)
    words.insert(position, number)

    while len(" ".join(words)) >= max_length and len(words) > 1:
        # drop the last word that is not the number
        drop = len(words) - 1
        if drop == position:
            drop -= 1
            position -= 1
        del words[drop]

    result = " ".join(words)[:max(max_length - 1, 1)]
    while len(result) < min_length:
        result += str(rand.choice(digits))
    return result
