from typing import Optional, Tuple, Union

import numpy as np
from torch.utils.data import DataLoader

from .datasets import DEFAULT_CACHE_DIR, RandomTextImageDataset, TransformedSubset
from .fonts import FontSet
from .transforms import image_transform, target_transform


def split_indices(n_samples: int, validation_split: Union[int, float], seed: int = 0):
    """
    Shuffle sample indices and split them into train and validation indices.

    :param validation_split: fraction of the samples, or a number of samples, held out for validation
    :return: (train indices, validation indices). Validation indices are None if nothing is held out.
    """
    if not validation_split:
        return np.arange(n_samples), None

    if isinstance(validation_split, int):
        assert validation_split > 0
        assert validation_split < n_samples, "validation set size is configured to be larger than entire dataset."
        len_valid = validation_split
    else:
        len_valid = int(n_samples * validation_split)

    # a local generator keeps the split reproducible without touching the global random state
    idx_full = np.random.default_rng(seed).permutation(n_samples)
    return idx_full[len_valid:], idx_full[:len_valid]


def make_train_valid_loaders(
    fonts: FontSet,
    random_seed: Optional[int] = None,
    total_samples: int = 10_000,
    text_length: Tuple[int, int] = (3, 8),
    text_image_dims: Tuple[int, int] = (128, 32),
    font_fingerprint_dims: Tuple[int, int] = (32, 32),
    cache_images: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
    random_augmentations: bool = True,
    validation_split: Union[int, float] = 0.0,
    split_seed: int = 0,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 1,
    pin_memory: bool = False,
):
    """
    Build one dataset and split it into a training loader and a validation loader.

    Random augmentations only apply to the training split. The validation split is read in a fixed order.

    :return: (train loader, validation loader). The validation loader is None if `validation_split` is 0.
    """
    dataset = RandomTextImageDataset(
        fonts,
        random_seed=random_seed,
        total_samples=total_samples,
        text_length=text_length,
        text_image_dims=text_image_dims,
        font_fingerprint_dims=font_fingerprint_dims,
        cache_images=cache_images,
        cache_dir=cache_dir,
    )
    train_idx, valid_idx = split_indices(len(dataset), validation_split, seed=split_seed)
    loader_kwargs = _loader_kwargs(batch_size, num_workers, pin_memory)

    train_set = TransformedSubset(
        dataset,
        None if valid_idx is None else train_idx,
        transform=image_transform(augment=random_augmentations),
        target_transform=target_transform(),
    )
    train_loader = DataLoader(train_set, shuffle=shuffle, **loader_kwargs)

    valid_loader = None
    if valid_idx is not None:
        valid_set = TransformedSubset(
            dataset,
            np.sort(valid_idx),
            transform=image_transform(augment=False),
            target_transform=target_transform(),
        )
        valid_loader = DataLoader(valid_set, shuffle=False, **loader_kwargs)

    return train_loader, valid_loader


def make_clustering_loader(
    fonts: FontSet,
    random_seed: Optional[int] = None,
    samples_per_font: int = 100,
    text_length: Tuple[int, int] = (3, 8),
    text_image_dims: Tuple[int, int] = (128, 32),
    cache_images: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
    batch_size: int = 128,
    num_workers: int = 1,
    pin_memory: bool = False,
):
    """
    Build a loader with `samples_per_font` clean images of every font and no targets.

    It is used to estimate the centre of each font in the latent space.
    """
    dataset = RandomTextImageDataset(
        fonts,
        random_seed=random_seed,
        total_samples=samples_per_font * len(fonts),
        text_length=text_length,
        text_image_dims=text_image_dims,
        group_by_font=True,
        return_target=False,
        cache_images=cache_images,
        cache_dir=cache_dir,
    )
    clustering_set = TransformedSubset(dataset, transform=image_transform(augment=False))
    return DataLoader(clustering_set, shuffle=False, **_loader_kwargs(batch_size, num_workers, pin_memory))


def _loader_kwargs(batch_size, num_workers, pin_memory):
    return {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        # workers stay alive between epochs, so fonts and caches are opened once per worker
        'persistent_workers': num_workers > 0,
    }
