from typing import Optional, Tuple, Union

import numpy as np
from torch.utils.data import DataLoader

from .corpus import TextCorpus
from .datasets import DEFAULT_CACHE_DIR, RandomTextImageDataset, TransformedSubset
from .fonts import FontSet
from .samplers import FontBalancedBatchSampler
from .transforms import image_transform, target_transform


_VALIDATION_SEED_OFFSET = 1_000_000_000


def split_fonts(fonts: FontSet, validation_split: Union[int, float], seed: int = 0):
    """
    Hold out whole font families for validation.

    The validation fonts never appear in training, so validation measures how the model does on unseen fonts.
    Families are kept together, because the weights of one typeface are nearly the same design.

    :param validation_split: fraction of the fonts, or a number of fonts, to hold out. Whole families are
        taken until that many fonts are reached, so the held out share can be somewhat larger.
    :return: (train font indices, validation font indices) as sorted positions in `fonts`. The validation
        indices are None if nothing is held out.
    """
    n_fonts = len(fonts)
    if not validation_split:
        return np.arange(n_fonts), None

    if isinstance(validation_split, int):
        assert validation_split > 0
        target = validation_split
    else:
        target = max(1, round(n_fonts * validation_split))

    families = {}
    for index, family in enumerate(fonts.families):
        families.setdefault(family, []).append(index)
    if len(families) < 2:
        raise ValueError("Holding out fonts for validation needs at least two font families.")

    # a local generator keeps the split reproducible without touching the global random state
    names = sorted(families)
    order = np.random.default_rng(seed).permutation(len(names))

    valid_idx = []
    for position in order[:-1]:  # at least one family stays for training
        if len(valid_idx) >= target:
            break
        valid_idx.extend(families[names[position]])

    valid_idx = np.sort(np.asarray(valid_idx))
    train_idx = np.setdiff1d(np.arange(n_fonts), valid_idx)
    return train_idx, valid_idx


def make_train_valid_loaders(
    fonts: FontSet,
    corpus: Optional[TextCorpus] = None,
    random_seed: Optional[int] = None,
    total_samples: int = 10_000,
    text_length: Tuple[int, int] = (3, 8),
    text_image_dims: Tuple[int, int] = (128, 32),
    font_fingerprint_dims: Tuple[int, int] = (32, 32),
    number_ratio: float = 0.0,
    render_scale: int = 2,
    cache_images: bool = True,
    cache_validation_images: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
    random_augmentations: bool = True,
    validation_augmentations: bool = False,
    capture_options: Optional[dict] = None,
    validation_split: Union[int, float] = 0.0,
    split_seed: int = 0,
    batch_size: int = 128,
    shuffle: bool = True,
    batch_fonts: Optional[int] = None,
    batch_samples_per_font: int = 4,
    epoch_samples: Optional[int] = None,
    validation_samples: Optional[int] = None,
    num_workers: int = 1,
    pin_memory: bool = False,
):
    """
    Build a training loader and a validation loader over disjoint sets of fonts.

    Synthetic variants of fonts are trained on, but not validated on.

    `validation_split` holds out whole font families, see `split_fonts`. `total_samples` is divided between
    the two datasets in proportion to their number of fonts.
    The validation set is read in a fixed order and has the same number of samples of every font.

    :param random_augmentations: put the training images through a simulated capture with a camera and its
        cleanup, see CaptureSimulation. Clean, tightly cropped images are used without it.
    :param validation_augmentations: do the same for the validation images, which measures how the model does
        on images like real ones. Every validation image gets the same distortion each time it is read.
    :param capture_options: arguments of CaptureSimulation
    :param batch_fonts: if given, training batches hold `batch_fonts` fonts with `batch_samples_per_font`
        samples each, as a contrastive loss needs. `batch_size` and `shuffle` are not used then.
    :param cache_images: keep the renderings of the training texts in a file under `cache_dir`. Without it
        every text is rendered when it is read, which gives the same images, needs no disk space and no time
        before the run, and so allows training sets of any size. The data loader workers then open the fonts
        themselves, which takes about a gigabyte of memory in each of them for a thousand fonts.
    :param cache_validation_images: the same for the validation texts. They are read again in every epoch,
        so a file is the better place for them.
    :param epoch_samples: number of training samples of an epoch, for a training set that is larger than one
        epoch. Every epoch then draws from its own part of the set, see FontBalancedBatchSampler. With
        `total_samples` of epochs times `epoch_samples`, no text is trained on twice. Needs `batch_fonts`.
    :param validation_samples: number of validation samples. Then all `total_samples` are for training, and
        the validation texts do not depend on the size of the training set. By default the validation set
        takes its share of `total_samples`.
    :return: (train loader, validation loader). The validation loader is None if `validation_split` is 0.
    """
    if random_seed is None:
        random_seed = int(np.random.randint(0, 2**31 - 1))

    train_fonts, valid_fonts = split_fonts(fonts, validation_split, seed=split_seed)
    if valid_fonts is not None:
        # validation measures real fonts only. The synthetic variants of held out fonts are not trained on
        # either, since they are in the family of their font: they only stand among the fonts to choose from.
        valid_fonts = np.array([index for index in valid_fonts if not fonts.is_synthetic(index)])
    if valid_fonts is None:
        n_valid, n_train = 0, total_samples
    elif validation_samples is not None:
        n_valid, n_train = int(validation_samples), total_samples
        # a fixed distance from the training seeds, so that the validation set is the same for every size
        # of the training set
        valid_seed = random_seed + _VALIDATION_SEED_OFFSET
    else:
        n_valid = max(1, round(total_samples * len(valid_fonts) / len(fonts)))
        n_train = total_samples - n_valid
        # seeds past those of the training samples, so validation does not repeat the training texts
        valid_seed = random_seed + n_train
    if n_train < 1 or (valid_fonts is not None and n_valid < 1):
        raise ValueError("total_samples is too small to leave samples for training and validation.")
    if epoch_samples is not None and batch_fonts is None:
        raise ValueError("epoch_samples needs batch_fonts, which sets how the samples of an epoch are drawn.")

    dataset_kwargs = {
        'corpus': corpus,
        'text_length': text_length,
        'text_image_dims': text_image_dims,
        'font_fingerprint_dims': font_fingerprint_dims,
        'number_ratio': number_ratio,
        'render_scale': render_scale,
        'cache_dir': cache_dir,
    }
    capture_options = capture_options or {}
    worker_kwargs = _worker_kwargs(num_workers, pin_memory)

    train_dataset = RandomTextImageDataset(
        fonts,
        font_indices=None if valid_fonts is None else train_fonts,
        random_seed=random_seed,
        total_samples=n_train,
        group_by_font=batch_fonts is not None,
        cache_images=cache_images,
        **dataset_kwargs,
    )
    train_set = TransformedSubset(
        train_dataset,
        transform=image_transform(text_image_dims, augment=random_augmentations, **capture_options),
        target_transform=target_transform(),
    )
    if batch_fonts is not None:
        batch_sampler = FontBalancedBatchSampler(
            train_dataset.sample_indices_by_font(), batch_fonts, batch_samples_per_font,
            samples_per_epoch=epoch_samples)
        train_loader = DataLoader(train_set, batch_sampler=batch_sampler, **worker_kwargs)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, **worker_kwargs)

    valid_loader = None
    if valid_fonts is not None:
        valid_dataset = RandomTextImageDataset(
            fonts,
            font_indices=valid_fonts,
            random_seed=valid_seed,
            total_samples=n_valid,
            group_by_font=True,
            cache_images=cache_validation_images,
            **dataset_kwargs,
        )
        valid_set = TransformedSubset(
            valid_dataset,
            transform=image_transform(text_image_dims, augment=validation_augmentations, **capture_options),
            target_transform=target_transform(),
            deterministic=True,
        )
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, **worker_kwargs)

    return train_loader, valid_loader


def make_clustering_loader(
    fonts: FontSet,
    corpus: Optional[TextCorpus] = None,
    random_seed: Optional[int] = None,
    samples_per_font: int = 100,
    text_length: Tuple[int, int] = (3, 8),
    text_image_dims: Tuple[int, int] = (128, 32),
    number_ratio: float = 0.0,
    render_scale: int = 2,
    cache_images: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
    random_augmentations: bool = False,
    capture_options: Optional[dict] = None,
    batch_size: int = 128,
    num_workers: int = 1,
    pin_memory: bool = False,
):
    """
    Build a loader with `samples_per_font` reference images of every font and no targets.

    It is used to estimate the centre of each font in the latent space. It covers all fonts, those held out
    for validation included: a held out font is identified among all fonts from these reference images alone.

    :param random_augmentations: put the reference images through the simulated capture as well, so that
        the centre of a font is estimated from images like the ones it is compared with. Every image gets
        the same distortion each time it is read.
    """
    dataset = RandomTextImageDataset(
        fonts,
        corpus=corpus,
        random_seed=random_seed,
        total_samples=samples_per_font * len(fonts),
        text_length=text_length,
        text_image_dims=text_image_dims,
        number_ratio=number_ratio,
        render_scale=render_scale,
        group_by_font=True,
        return_target=False,
        cache_images=cache_images,
        cache_dir=cache_dir,
    )
    clustering_set = TransformedSubset(
        dataset,
        transform=image_transform(text_image_dims, augment=random_augmentations, **(capture_options or {})),
        deterministic=True,
    )
    return DataLoader(
        clustering_set, batch_size=batch_size, shuffle=False, **_worker_kwargs(num_workers, pin_memory))


def _worker_kwargs(num_workers, pin_memory):
    return {
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        # workers stay alive between epochs, so fonts and caches are opened once per worker
        'persistent_workers': num_workers > 0,
    }
