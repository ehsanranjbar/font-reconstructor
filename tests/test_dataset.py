import pickle

import numpy as np
import torch

from font_reconstructor.dataset import (FontSet, RandomTextImageDataset, make_clustering_loader,
                                        make_train_valid_loaders, split_indices)


def make_dataset(fonts, tmp_path, **kwargs):
    kwargs.setdefault('random_seed', 7)
    kwargs.setdefault('total_samples', 24)
    return RandomTextImageDataset(fonts, cache_dir=str(tmp_path / 'cache'), **kwargs)


def test_fontset(fonts):
    assert len(fonts) == 3
    assert fonts.num_glyphs == 8
    assert fonts.glyphs(1) == 'abcdefgh'


def test_fontset_drops_fonts_that_fail_to_open(fonts_dir, tmp_path):
    root, _ = fonts_dir
    annotation_file = tmp_path / 'fonts.csv'
    annotation_file.write_text(
        'font,file,supported_charset\n'
        'Missing,missing.ttf,a b\n'
        'DejaVuSerif,DejaVuSerif.ttf,a b\n'
    )
    fonts = FontSet(str(root), str(annotation_file))
    assert fonts.names == ['DejaVuSerif']

    # the remaining font is addressed by its position, not by its row in the annotation file
    dataset = make_dataset(fonts, tmp_path, total_samples=4)
    assert dataset[0]['target'].shape == (32, 32, 2)
    assert dataset[0]['target'].any()


def test_sample_layout(fonts, tmp_path):
    sample = make_dataset(fonts, tmp_path)[0]
    assert sample['image'].shape == (32, 128) and sample['image'].dtype == np.uint8
    assert sample['target'].shape == (32, 32, 8) and sample['target'].dtype == np.uint8
    assert sample['font'] == fonts.names[sample['font_index']]
    assert 3 <= len(sample['text']) < 8


def test_null_characters_leave_their_channel_empty(fonts, tmp_path):
    dataset = make_dataset(fonts, tmp_path)
    fingerprint = dataset.generate_font_fingerprint(0)
    assert fingerprint[:, :, :6].any(axis=(0, 1)).all()
    assert not fingerprint[:, :, 6:].any()


def test_samples_are_deterministic_and_match_the_cache(fonts, tmp_path):
    cached = make_dataset(fonts, tmp_path)
    reloaded = make_dataset(fonts, tmp_path)
    rendered = make_dataset(fonts, tmp_path, cache_images=False, cache_fingerprints=False)

    for idx in range(len(cached)):
        for other in (reloaded, rendered):
            assert cached[idx]['text'] == other[idx]['text']
            assert cached[idx]['font_index'] == other[idx]['font_index']
            np.testing.assert_array_equal(cached[idx]['image'], other[idx]['image'])
            np.testing.assert_array_equal(cached[idx]['target'], other[idx]['target'])


def test_random_seed_default(fonts, tmp_path):
    dataset = make_dataset(fonts, tmp_path, random_seed=None, cache_images=False)
    assert isinstance(dataset.random_seed, int)
    assert dataset[0]['image'].shape == (32, 128)


def test_cache_key_follows_the_settings(fonts, tmp_path):
    def cache_files():
        return {path.name for path in (tmp_path / 'cache').iterdir()}

    make_dataset(fonts, tmp_path)
    first = cache_files()
    assert len(first) == 2

    make_dataset(fonts, tmp_path)
    assert cache_files() == first

    dataset = make_dataset(fonts, tmp_path, font_fingerprint_dims=(16, 16))
    assert dataset[0]['target'].shape == (16, 16, 8)
    assert len(cache_files()) == 3

    make_dataset(fonts, tmp_path, random_seed=8)
    assert len(cache_files()) == 4


def test_dataset_pickles_without_its_data(fonts, tmp_path):
    dataset = make_dataset(fonts, tmp_path)
    expected = dataset[5]

    payload = pickle.dumps(dataset)
    assert len(payload) < dataset[0]['target'].nbytes

    restored = pickle.loads(payload)
    np.testing.assert_array_equal(restored[5]['image'], expected['image'])
    np.testing.assert_array_equal(restored[5]['target'], expected['target'])
    uncached = pickle.loads(pickle.dumps(make_dataset(fonts, tmp_path, cache_images=False)))
    np.testing.assert_array_equal(uncached[5]['image'], expected['image'])


def test_split_indices():
    train_idx, valid_idx = split_indices(100, 0.1)
    assert len(train_idx) == 90 and len(valid_idx) == 10
    assert set(train_idx).isdisjoint(valid_idx)
    np.testing.assert_array_equal(split_indices(100, 0.1)[1], valid_idx)

    state = np.random.get_state()[1].copy()
    split_indices(100, 0.1)
    np.testing.assert_array_equal(np.random.get_state()[1], state)

    train_idx, valid_idx = split_indices(100, 0.0)
    assert len(train_idx) == 100 and valid_idx is None


def test_train_valid_loaders(fonts, tmp_path):
    train_loader, valid_loader = make_train_valid_loaders(
        fonts, random_seed=7, total_samples=40, validation_split=0.25, batch_size=8, num_workers=0,
        cache_dir=str(tmp_path / 'cache'),
    )
    assert len(train_loader.dataset) == 30 and len(valid_loader.dataset) == 10

    batch = next(iter(train_loader))
    assert batch['image'].shape == (8, 1, 32, 128)
    assert batch['target'].shape == (8, 8, 32, 32)
    assert batch['font_index'].dtype == torch.int64
    assert -1 <= batch['image'].min() and batch['image'].max() <= 1

    # validation images are not augmented, so reading them twice gives the same tensors
    first = torch.cat([batch['image'] for batch in valid_loader])
    second = torch.cat([batch['image'] for batch in valid_loader])
    assert torch.equal(first, second)

    # training images are augmented
    assert not torch.equal(train_loader.dataset[0]['image'], train_loader.dataset[0]['image'])


def test_clustering_loader_covers_every_font_equally(fonts, tmp_path):
    loader = make_clustering_loader(
        fonts, random_seed=3, samples_per_font=5, batch_size=4, num_workers=0, cache_dir=str(tmp_path / 'cache'),
    )
    batches = list(loader)
    assert all('target' not in batch for batch in batches)
    font_index = torch.cat([batch['font_index'] for batch in batches])
    assert torch.bincount(font_index).tolist() == [5, 5, 5]
