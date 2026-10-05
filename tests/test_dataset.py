import collections
import pickle

import numpy as np
import pytest
import torch
from PIL import features

from font_reconstructor.dataset import (CaptureSimulation, FontBalancedBatchSampler, FontSet, RandomTextImageDataset, Variant,
                                        TextCorpus, clean_text_image, derive_style, fit_to_box, image_transform,
                                        make_clustering_loader, make_train_valid_loaders, normalize_text,
                                        resolve_layout_engine, split_fonts)
from font_reconstructor.dataset.datasets import _add_number


def make_dataset(fonts, tmp_path, **kwargs):
    kwargs.setdefault('random_seed', 7)
    kwargs.setdefault('total_samples', 24)
    return RandomTextImageDataset(fonts, cache_dir=str(tmp_path / 'cache'), **kwargs)


def test_fontset(fonts):
    assert len(fonts) == 3
    assert fonts.num_glyphs == 8
    assert fonts.glyphs(1) == 'abcdefgh'
    # without a family column the family is read from the font files
    assert fonts.families == ['DejaVu Sans', 'DejaVu Serif', 'DejaVu Sans Mono']


def test_fontset_family_column(fonts_dir, tmp_path):
    root, _ = fonts_dir
    annotation_file = tmp_path / 'fonts.csv'
    annotation_file.write_text(
        'font,file,supported_charset,family\n'
        'Sans,DejaVuSans.ttf,a b,DejaVu\n'
        'Serif,DejaVuSerif.ttf,a b,DejaVu\n'
        'Mono,DejaVuSansMono.ttf,a b,\n'
    )
    fonts = FontSet(str(root), str(annotation_file))
    assert fonts.families == ['DejaVu', 'DejaVu', 'DejaVu Sans Mono']


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


def test_layout_engine(fonts_dir):
    root, annotation_file = fonts_dir
    has_raqm = bool(features.check('raqm'))

    assert resolve_layout_engine('basic') == 'basic'
    assert resolve_layout_engine('auto') == ('raqm' if has_raqm else 'basic')
    with pytest.raises(ValueError):
        resolve_layout_engine('pango')

    if has_raqm:
        assert FontSet(str(root), str(annotation_file), layout_engine='raqm').layout_engine == 'raqm'
    else:
        # asking for shaping that is not available has to fail, not silently draw unjoined letters
        with pytest.raises(RuntimeError, match='raqm'):
            FontSet(str(root), str(annotation_file), layout_engine='raqm')

    # the engine changes what is rendered, so it is part of the cache signature
    basic = FontSet(str(root), str(annotation_file), layout_engine='basic')
    assert 'basic' in basic.signature()


def test_sample_layout(fonts, tmp_path):
    sample = make_dataset(fonts, tmp_path)[0]
    # the text is rendered at twice the size of the model input, with a border around it
    assert sample['image'].shape == (64, 256) and sample['image'].dtype == np.uint8
    assert not sample['image'][:4].any() and not sample['image'][:, :16].any()
    assert make_dataset(fonts, tmp_path, render_scale=1, cache_images=False)[0]['image'].shape == (32, 128)
    assert sample['target'].shape == (32, 32, 8) and sample['target'].dtype == np.uint8
    assert sample['font'] == fonts.names[sample['font_index']]
    assert fonts.style_names[sample['style_index']] == fonts.styles[sample['font_index']]
    assert 3 <= len(sample['text']) < 8

    # the glyphs of the fingerprint that the text shows
    assert sample['seen'].shape == (8,) and sample['seen'].dtype == bool
    glyphs = fonts.glyphs(sample['font_index'])
    assert [glyph in sample['text'] for glyph in glyphs] == sample['seen'][:len(glyphs)].tolist()
    assert sample['seen'].any()
    dataset = make_dataset(fonts, tmp_path)
    # font 0 leaves two glyph slots empty, they are never seen
    assert dataset.seen_glyphs(0, 'abh').tolist() == [True, True] + [False] * 6
    assert 'seen' not in make_dataset(fonts, tmp_path, return_target=False, cache_images=False)[0]


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
    assert dataset[0]['image'].shape == (64, 256)


def test_cache_key_follows_the_settings(fonts, tmp_path, corpus_file):
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

    # a subset of the fonts and a text corpus render other images, the fingerprints stay shared
    make_dataset(fonts, tmp_path, font_indices=[0, 2])
    assert len(cache_files()) == 5

    make_dataset(fonts, tmp_path, corpus=TextCorpus(corpus_file))
    assert len(cache_files()) == 6

    make_dataset(fonts, tmp_path, render_scale=1)
    assert len(cache_files()) == 7

    make_dataset(fonts, tmp_path, number_ratio=0.5)
    assert len(cache_files()) == 8


def test_dataset_pickles_without_its_data(fonts, tmp_path, corpus_file):
    dataset = make_dataset(fonts, tmp_path, corpus=TextCorpus(corpus_file))
    expected = dataset[5]

    payload = pickle.dumps(dataset)
    assert len(payload) < dataset[0]['target'].nbytes

    restored = pickle.loads(payload)
    assert restored[5]['text'] == expected['text']
    np.testing.assert_array_equal(restored[5]['image'], expected['image'])
    np.testing.assert_array_equal(restored[5]['target'], expected['target'])
    uncached = pickle.loads(pickle.dumps(make_dataset(fonts, tmp_path, corpus=TextCorpus(corpus_file),
                                                      cache_images=False)))
    np.testing.assert_array_equal(uncached[5]['image'], expected['image'])


def test_font_indices_restrict_the_fonts(fonts, tmp_path):
    random_fonts = make_dataset(fonts, tmp_path, font_indices=[0, 2], total_samples=40, cache_images=False)
    assert {random_fonts[idx]['font_index'] for idx in range(40)} == {0, 2}

    grouped = make_dataset(fonts, tmp_path, font_indices=[2, 1], total_samples=7, group_by_font=True,
                           cache_images=False)
    assert [grouped[idx]['font_index'] for idx in range(7)] == [2, 1, 2, 1, 2, 1, 2]
    assert [list(indices) for indices in grouped.sample_indices_by_font()] == [[0, 2, 4, 6], [1, 3, 5]]

    with pytest.raises(ValueError):
        random_fonts.sample_indices_by_font()


def test_normalize_text():
    # arabic yeh and kaf become persian, diacritics and tatweel are dropped, digits become persian digits
    assert normalize_text('كتابي') == 'کتابی'
    assert normalize_text('مَـن') == 'من'
    assert normalize_text('۱۲ 34 ٥') == '۱۲ ۳۴ ۵'
    # punctuation and the zero width non-joiner separate words
    assert normalize_text('می‌روم، خانه!') == 'می روم خانه'


def test_corpus_samples_real_words(corpus_file):
    corpus = TextCorpus(corpus_file)
    charset = 'abcdefgh'
    words = {word for line in corpus.lines_for(charset) for word in line}
    # 'A' and 'café' have characters outside the charset, punctuation only separates words
    assert words == {'bad', 'cab', 'faded', 'head', 'had', 'a', 'bag', 'feed', 'deaf', 'bee', 'bead',
                     'abcdefgh', 'dab'}

    rand = np.random.RandomState(0)
    texts = [corpus.sample(rand, (3, 8), charset) for _ in range(300)]
    assert all(text is not None for text in texts)
    assert all(3 <= len(text) < 8 for text in texts)
    assert all(set(text.split()) <= words for text in texts)
    assert any(' ' in text for text in texts)

    # the same random state gives the same texts
    rand = np.random.RandomState(0)
    assert [corpus.sample(rand, (3, 8), charset) for _ in range(300)] == texts

    # words with characters outside the charset are not used
    restricted = corpus.lines_for('abcdef')
    assert 'head' not in {word for line in restricted for word in line}
    # no text of the corpus fits, the caller falls back to random characters
    assert corpus.sample(np.random.RandomState(0), (3, 8), 'xyz') is None
    assert corpus.sample(np.random.RandomState(0), (30, 40), 'abcdefgh') is None


def test_dataset_draws_texts_from_the_corpus(fonts, tmp_path, corpus_file):
    corpus = TextCorpus(corpus_file)
    dataset = make_dataset(fonts, tmp_path, corpus=corpus, total_samples=60, cache_images=False)
    known = {word for line in corpus.lines_for('abcdefgh') for word in line}

    for idx in range(60):
        sample = dataset[idx]
        assert set(sample['text'].split()) <= known
        # the first font has no g and h, so none of its texts uses them
        if sample['font_index'] == 0:
            assert not set('gh') & set(sample['text'])


def test_split_fonts(fonts):
    train_idx, valid_idx = split_fonts(fonts, 0.34)
    assert len(train_idx) == 2 and len(valid_idx) == 1
    assert set(train_idx).isdisjoint(valid_idx)
    np.testing.assert_array_equal(split_fonts(fonts, 0.34)[1], valid_idx)

    # the split does not touch the global random state
    state = np.random.get_state()[1].copy()
    split_fonts(fonts, 0.34)
    np.testing.assert_array_equal(np.random.get_state()[1], state)

    train_idx, valid_idx = split_fonts(fonts, 0.0)
    assert len(train_idx) == 3 and valid_idx is None

    # one family always stays for training
    train_idx, valid_idx = split_fonts(fonts, 0.99)
    assert len(train_idx) == 1 and len(valid_idx) == 2


def test_split_fonts_keeps_families_together(fonts):
    fonts.families = ['one', 'two', 'one']
    for seed in range(5):
        train_idx, valid_idx = split_fonts(fonts, 1, seed=seed)
        assert sorted(valid_idx.tolist()) in ([0, 2], [1])

    fonts.families = ['one', 'one', 'one']
    with pytest.raises(ValueError):
        split_fonts(fonts, 0.5)


def test_font_balanced_batches():
    indices_by_font = [np.arange(start, 60, 6) for start in range(6)]
    sampler = FontBalancedBatchSampler(indices_by_font, fonts_per_batch=4, samples_per_font=3)
    assert len(sampler) == 60 // 12

    torch.manual_seed(0)
    batches = list(sampler)
    assert len(batches) == 5
    for batch in batches:
        fonts_in_batch = collections.Counter(index % 6 for index in batch)
        assert len(batch) == 12 and len(set(batch)) == 12
        assert sorted(fonts_in_batch.values()) == [3, 3, 3, 3]

    # the torch seed fixes the batches
    torch.manual_seed(0)
    assert list(sampler) == batches

    # fewer fonts than asked for
    few = FontBalancedBatchSampler(indices_by_font[:2], fonts_per_batch=4, samples_per_font=3)
    assert few.batch_size == 6

    with pytest.raises(ValueError):
        FontBalancedBatchSampler(indices_by_font, fonts_per_batch=4, samples_per_font=1)


def test_epochs_can_draw_from_their_own_part_of_the_samples():
    # six fonts with forty samples each, as ranges like those of a dataset
    indices_by_font = [range(start, 240, 6) for start in range(6)]
    sampler = FontBalancedBatchSampler(indices_by_font, fonts_per_batch=3, samples_per_font=4, samples_per_epoch=60)
    # the dataset holds four epochs of sixty samples, in batches of twelve
    assert sampler.parts == 4 and len(sampler) == 5

    torch.manual_seed(0)
    seen = []
    for epoch in range(1, 5):
        sampler.set_epoch(epoch)
        batches = list(sampler)
        assert len(batches) == 5 and all(len(batch) == 12 for batch in batches)
        for batch in batches:
            assert sorted(collections.Counter(index % 6 for index in batch).values()) == [4, 4, 4]
        seen.append({index for batch in batches for index in batch})
    # no sample is drawn in two epochs, and every epoch stays within its tenth to twentieth sample of a font
    for first in range(4):
        for second in range(first + 1, 4):
            assert not seen[first] & seen[second]
        assert all(first * 10 <= index // 6 < (first + 1) * 10 for index in seen[first])

    # a font goes through all its samples of the part before one of them comes again
    one_font = FontBalancedBatchSampler([range(0, 40)], fonts_per_batch=1, samples_per_font=2, samples_per_epoch=10)
    one_font.set_epoch(2)
    assert sorted(index for batch in one_font for index in batch) == list(range(10, 20))

    # after the last part the first one comes again, and the epoch is what decides, not the number of passes
    sampler.set_epoch(5)
    assert {index for batch in sampler for index in batch} <= {index for index in range(240) if index // 6 < 10}
    sampler.set_epoch(2)
    torch.manual_seed(3)
    again = list(sampler)
    torch.manual_seed(3)
    assert list(sampler) == again

    with pytest.raises(ValueError):
        FontBalancedBatchSampler(indices_by_font, 3, 4, batches_per_epoch=2, samples_per_epoch=60)
    with pytest.raises(ValueError):
        FontBalancedBatchSampler(indices_by_font, 3, 4, samples_per_epoch=1000)
    with pytest.raises(ValueError, match='fewer samples'):
        FontBalancedBatchSampler([range(0, 3), range(3, 240)], 2, 2, samples_per_epoch=10)


def test_rendering_when_read_gives_the_cached_images(fonts, corpus_file, tmp_path):
    corpus = TextCorpus([str(corpus_file)])
    settings = dict(random_seed=7, total_samples=60, validation_split=0.34, batch_fonts=2, batch_samples_per_font=2,
                    num_workers=0, corpus=corpus, number_ratio=0.3, random_augmentations=False)
    cache_dir = tmp_path / 'cache'
    cached_train, cached_valid = make_train_valid_loaders(fonts, cache_dir=str(cache_dir), **settings)
    assert len(list(cache_dir.glob('images_*.npy'))) == 2

    live_dir = tmp_path / 'live'
    live_train, live_valid = make_train_valid_loaders(
        fonts, cache_dir=str(live_dir), cache_images=False, cache_validation_images=False, **settings)
    # only the fingerprints are kept in a file
    assert [path.name.split('_')[0] for path in live_dir.iterdir()] == ['fingerprints']

    for cached, live in ((cached_train, live_train), (cached_valid, live_valid)):
        assert len(cached.dataset) == len(live.dataset)
        for index in range(len(cached.dataset)):
            first, second = cached.dataset[index], live.dataset[index]
            assert first['text'] == second['text'] and first['font_index'] == second['font_index']
            assert torch.equal(first['image'], second['image'])
            assert torch.equal(second['image'], live.dataset[index]['image'])

    # by default the validation images are still kept in a file, they are read in every epoch
    mixed_dir = tmp_path / 'mixed'
    make_train_valid_loaders(fonts, cache_dir=str(mixed_dir), cache_images=False, **settings)
    assert sorted(path.name.split('_')[0] for path in mixed_dir.iterdir()) == ['fingerprints', 'images']


def test_training_set_larger_than_an_epoch(fonts, tmp_path):
    settings = dict(random_seed=7, validation_split=0.34, batch_fonts=2, batch_samples_per_font=2, num_workers=0,
                    cache_dir=str(tmp_path / 'cache'), cache_images=False)
    train_loader, valid_loader = make_train_valid_loaders(
        fonts, total_samples=120, epoch_samples=40, validation_samples=10, **settings)
    # all samples are for training, and an epoch is a third of them
    assert len(train_loader.dataset) == 120 and len(valid_loader.dataset) == 10
    assert len(train_loader) == 10 and train_loader.batch_sampler.parts == 3

    def texts(loader):
        return [text for batch in loader for text in batch['text']]

    # the validation set does not depend on the size of the training set
    _, other_valid = make_train_valid_loaders(fonts, total_samples=400, epoch_samples=40, validation_samples=10, **settings)
    assert texts(other_valid) == texts(valid_loader)

    train_loader.batch_sampler.set_epoch(1)
    first = {int(index) for batch in train_loader.batch_sampler for index in batch}
    train_loader.batch_sampler.set_epoch(2)
    assert not first & {int(index) for batch in train_loader.batch_sampler for index in batch}

    with pytest.raises(ValueError, match='batch_fonts'):
        make_train_valid_loaders(fonts, total_samples=120, epoch_samples=40, validation_split=0.34, num_workers=0,
                                 cache_dir=str(tmp_path / 'cache'))


def test_train_valid_loaders_hold_out_fonts(fonts, tmp_path):
    train_loader, valid_loader = make_train_valid_loaders(
        fonts, random_seed=7, total_samples=60, validation_split=0.34, batch_size=8, num_workers=0,
        cache_dir=str(tmp_path / 'cache'),
    )
    assert len(train_loader.dataset) == 40 and len(valid_loader.dataset) == 20

    train_fonts = {int(i) for batch in train_loader for i in batch['font_index']}
    valid_fonts = {int(i) for batch in valid_loader for i in batch['font_index']}
    assert len(train_fonts) == 2 and len(valid_fonts) == 1
    assert train_fonts.isdisjoint(valid_fonts)

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


def test_validation_augmentations_are_the_same_on_every_read(fonts, tmp_path):
    def valid_images(**kwargs):
        _, valid_loader = make_train_valid_loaders(
            fonts, random_seed=7, total_samples=60, validation_split=0.34, batch_size=8, num_workers=0,
            cache_dir=str(tmp_path / 'cache'), **kwargs)
        return torch.cat([batch['image'] for batch in valid_loader])

    clean = valid_images()
    captured = valid_images(validation_augmentations=True)
    assert captured.shape == clean.shape
    assert not torch.equal(captured, clean)
    assert torch.equal(captured, valid_images(validation_augmentations=True))

    # the capture settings of the config reach the simulation
    mild = valid_images(validation_augmentations=True, capture_options={'max_rotation': 0.0, 'binarize_prob': 1.0})
    assert not torch.equal(mild, captured)
    assert set(mild.unique().tolist()) - {-1.0, 1.0}, 'resizing a binary image leaves grey edges'


def test_train_loader_with_font_balanced_batches(fonts, tmp_path):
    train_loader, valid_loader = make_train_valid_loaders(
        fonts, random_seed=7, total_samples=60, batch_fonts=3, batch_samples_per_font=4, num_workers=0,
        cache_dir=str(tmp_path / 'cache'),
    )
    assert valid_loader is None
    assert len(train_loader) == 60 // 12

    for batch in train_loader:
        assert batch['image'].shape == (12, 1, 32, 128)
        assert torch.bincount(batch['font_index']).tolist() == [4, 4, 4]


def test_clustering_loader_covers_every_font_equally(fonts, tmp_path):
    loader = make_clustering_loader(
        fonts, random_seed=3, samples_per_font=5, batch_size=4, num_workers=0, cache_dir=str(tmp_path / 'cache'),
    )
    batches = list(loader)
    assert all('target' not in batch for batch in batches)
    font_index = torch.cat([batch['font_index'] for batch in batches])
    assert torch.bincount(font_index).tolist() == [5, 5, 5]

    captured = make_clustering_loader(
        fonts, random_seed=3, samples_per_font=5, batch_size=4, num_workers=0, cache_dir=str(tmp_path / 'cache'),
        random_augmentations=True,
    )
    first = torch.cat([batch['image'] for batch in captured])
    assert not torch.equal(first, torch.cat([batch['image'] for batch in batches]))
    assert torch.equal(first, torch.cat([batch['image'] for batch in captured]))


def text_canvas(fonts, text='abcdef', dims=(256, 64)):
    from font_reconstructor.dataset import render_text
    return render_text(fonts.ttf(1), text, dims, fill=0.85)


def test_fit_to_box_crops_to_the_text():
    from PIL import Image

    image = np.zeros((64, 256), dtype=np.uint8)
    image[20:40, 100:140] = 255  # a 40 wide, 20 high block of "text"

    fitted = np.asarray(fit_to_box(Image.fromarray(image), (128, 32)))
    rows, cols = np.where(fitted > 127)
    # the block is scaled up until it reaches the top and bottom of the box, and centred
    assert fitted.shape == (32, 128)
    assert rows.min() <= 2 and rows.max() >= 29
    assert abs((cols.min() + cols.max()) / 2 - 63.5) <= 1.5
    assert 54 <= cols.max() - cols.min() + 1 <= 62

    # margins leave background around the text, alignment moves it to a side
    loose = np.asarray(fit_to_box(Image.fromarray(image), (128, 32), margins=(0, 0.5, 0, 0.5), align=(0.0, 0.5)))
    rows, cols = np.where(loose > 127)
    assert rows.min() >= 6 and rows.max() <= 25
    assert cols.min() <= 1

    assert not np.asarray(fit_to_box(Image.fromarray(np.zeros((64, 256), dtype=np.uint8)), (128, 32))).any()
    np.testing.assert_array_equal(clean_text_image(image, (128, 32)), fitted)


def test_render_text_keeps_ink_that_starts_before_the_origin(fonts):
    from PIL import Image, ImageDraw
    from font_reconstructor.dataset import render_text

    found_overhang = False
    for index in range(len(fonts)):
        ttf = fonts.ttf(index)
        for text in ('j', 'f', 'jab', 'gab'):
            left, top, right, bottom = ttf.getbbox(text, anchor='lt')
            found_overhang |= left < 0 or top < 0

            # the same text drawn with room on all sides, cut to its bounding box
            pad = 40
            reference = Image.new('L', (right - left + 2 * pad, bottom - top + 2 * pad), 0)
            ImageDraw.Draw(reference).text((pad, pad), text, fill=255, anchor='lt', font=ttf)
            reference = np.asarray(reference)[pad + top:pad + bottom, pad + left:pad + right]

            rendered = render_text(ttf, text, (right - left, bottom - top))
            np.testing.assert_array_equal(rendered, reference)
    assert found_overhang, 'the test fonts should have a letter that reaches left of its origin'


def test_crop_keeps_thin_and_faint_strokes():
    from PIL import Image
    from font_reconstructor.dataset.capture import ink_bbox

    image = np.zeros((64, 256), dtype=np.uint8)
    image[20:40, 100:140] = 255   # a letter
    image[46:47, 110:130] = 110   # a faint hairline below it, like the tail or the dots of a letter
    image[5, 5] = 255             # a single bright pixel of noise

    left, top, right, bottom = ink_bbox(Image.fromarray(image))
    # the hairline belongs to the text, the noise does not
    assert bottom >= 47 and top <= 20 and left <= 100 and right >= 140
    assert top > 6

    fitted = np.asarray(fit_to_box(Image.fromarray(image), (128, 32)))
    assert fitted[-4:].max() > 30, 'the hairline is at the bottom of the crop'

    # by default the random crop of the capture simulation leaves a margin, it never cuts into the text
    assert CaptureSimulation((128, 32)).margin[0] >= 0


def test_capture_simulation(fonts):
    canvas = text_canvas(fonts)
    simulation = CaptureSimulation((128, 32))

    first = simulation(canvas, np.random.default_rng(1))
    assert first.shape == (32, 128) and first.dtype == np.uint8
    np.testing.assert_array_equal(first, simulation(canvas, np.random.default_rng(1)))
    assert not np.array_equal(first, simulation(canvas, np.random.default_rng(2)))
    assert not np.array_equal(first, clean_text_image(canvas, (128, 32)))

    # the text survives: every result has bright strokes on a mostly dark background
    results = np.stack([simulation(canvas, np.random.default_rng(seed)) for seed in range(100)])
    ink_share = (results > 127).mean(axis=(1, 2))
    assert (ink_share > 0.01).mean() > 0.95
    assert np.median(ink_share) < 0.5

    # binarization leaves two grey levels before the final resize, the grey path keeps many
    binary = CaptureSimulation((128, 32), binarize_prob=1.0)._clean_up
    from PIL import Image
    photo = Image.fromarray(255 - (canvas // 2))
    assert set(np.unique(np.asarray(binary(photo, np.random.default_rng(0))))) <= {0, 255}
    grey = CaptureSimulation((128, 32), binarize_prob=0.0)._clean_up
    assert len(np.unique(np.asarray(grey(photo, np.random.default_rng(0))))) > 2


def test_image_transform(fonts):
    canvas = text_canvas(fonts)

    clean = image_transform((128, 32))
    tensor = clean(canvas)
    assert tensor.shape == (1, 32, 128) and tensor.dtype == torch.float32
    assert tensor.min() == -1 and tensor.max() > 0.9
    assert torch.equal(tensor, clean(canvas, seed=5))

    captured = image_transform((128, 32), augment=True)
    assert torch.equal(captured(canvas, seed=3), captured(canvas, seed=3))
    assert not torch.equal(captured(canvas, seed=3), captured(canvas, seed=4))
    # without a seed it follows the random state of torch
    torch.manual_seed(0)
    first = captured(canvas)
    assert not torch.equal(first, captured(canvas))
    torch.manual_seed(0)
    assert torch.equal(first, captured(canvas))


def test_add_number():
    charset = 'abc012'
    for seed in range(200):
        rand = np.random.RandomState(seed)
        text = _add_number(rand, 'ab cab', (3, 8), charset)
        assert 3 <= len(text) < 8
        assert any(char.isdigit() for char in text)
        assert set(text) <= set(charset) | {' '}

    # a charset without digits leaves the text alone
    assert _add_number(np.random.RandomState(0), 'ab cab', (3, 8), 'abc') == 'ab cab'


def test_number_ratio_adds_numbers_to_texts(fonts, tmp_path):
    fonts.charsets = [charset + ' 1 2' for charset in fonts.charsets]
    options = dict(total_samples=80, cache_images=False, return_target=False)
    plain = make_dataset(fonts, tmp_path, **options)
    numbered = make_dataset(fonts, tmp_path, number_ratio=1.0, **options)

    def share_with_digits(dataset):
        return np.mean([any(char.isdigit() for char in dataset[idx]['text']) for idx in range(80)])

    # random texts hold a digit now and then, with number_ratio 1 every text holds a number
    assert share_with_digits(numbered) == 1.0
    assert share_with_digits(plain) < 1.0
    assert all(3 <= len(numbered[idx]['text']) < 8 for idx in range(80))


@pytest.mark.parametrize('names, style', [
    (('Vazirmatn Bold', 'Bold'), 'bold'),
    (('Vazirmatn ExtraLight', 'ExtraLight'), 'light'),
    (('Sahel FD SemiBold', 'SemiBold'), 'bold'),
    (('Samim FD Medium', 'Medium'), 'regular'),
    (('S_HANI REGULAR', 'REGULAR'), 'regular'),
    (('Amiri Italic', 'Italic'), 'italic'),
    (('B Mashhad BoldItalic', 'BoldItalic'), 'bold italic'),
    (('X Nimrooz ItalicR', 'ItalicR'), 'italic'),
    # decorations are often only named in the display name, and they outrank the weight
    (('X Kamran Outline', 'Bold'), 'outline'),
    (('2 Niki Border', ' '), 'outline'),
    (('X Majid Shadow', 'Shadow'), 'shadow'),
    # style words only count as whole words
    (('Delight', 'Regular'), 'regular'),
])
def test_derive_style(names, style):
    assert derive_style(*names) == style


def test_fontset_styles(fonts, fonts_dir, tmp_path):
    # the fonts of the tests are all regular
    assert fonts.styles == ['regular', 'regular', 'regular']
    assert fonts.style_names == ['regular'] and fonts.style_indices == [0, 0, 0]

    root, _ = fonts_dir
    annotation_file = tmp_path / 'fonts.csv'
    annotation_file.write_text(
        'font,file,supported_charset,style\n'
        'Sans,DejaVuSans.ttf,a b,naskh\n'
        'Serif Bold,DejaVuSerif.ttf,a b,\n'
        'Mono,DejaVuSansMono.ttf,a b,kufi\n'
    )
    labelled = FontSet(str(root), str(annotation_file))
    # the column wins, fonts without a label fall back to the style derived from their names
    assert labelled.styles == ['naskh', 'bold', 'kufi']
    assert labelled.style_names == ['bold', 'kufi', 'naskh']
    assert labelled.style_indices == [2, 0, 1]


def test_background_estimate_ignores_heavy_text():
    from font_reconstructor.dataset.capture import _estimate_background

    # paper that gets brighter to the right, with a block of ink far wider than any stroke
    x = np.linspace(150, 230, 256, dtype=np.float32)[None, :].repeat(64, axis=0)
    photo = x.copy()
    photo[12:52, 60:200] = 30
    background = _estimate_background(photo)
    # the paper level is recovered under the ink too, so the ink is not mistaken for paper
    assert np.abs(background - x).max() < 12
    assert (background[12:52, 60:200] - photo[12:52, 60:200]).min() > 100


def test_cleanup_keeps_heavy_strokes_solid():
    from PIL import Image

    # a rendering with a stroke 36 pixels thick, as the heaviest fonts have
    canvas = np.zeros((64, 256), dtype=np.uint8)
    canvas[14:50, 40:210] = 255
    for binarize in (0.0, 1.0):
        simulation = CaptureSimulation((128, 32), binarize_prob=binarize)
        for seed in range(20):
            rng = np.random.default_rng(seed)
            photo = simulation._photograph(Image.fromarray(canvas), rng)
            cleaned = np.asarray(simulation._clean_up(photo, rng), dtype=np.float32)
            # the middle of the stroke is ink, not only its edges
            rows, columns = cleaned.shape
            middle = cleaned[int(rows * 0.4):int(rows * 0.6), int(columns * 0.3):int(columns * 0.7)]
            assert np.median(middle) > 150, (binarize, seed)
            assert np.median(cleaned[:3]) < 60, 'the paper above the stroke stays dark'


def test_text_agreement_tells_text_from_damage(fonts):
    from PIL import Image, ImageFilter
    from font_reconstructor.dataset.capture import text_agreement

    text = Image.fromarray(text_canvas(fonts))
    assert text_agreement(text, text) == pytest.approx(1.0)
    # what a capture does to text that survives it: softer, smaller, thicker
    assert text_agreement(text, text.filter(ImageFilter.GaussianBlur(1.5))) > 0.9
    assert text_agreement(text, text.resize((128, 32))) > 0.9
    assert text_agreement(text, text.filter(ImageFilter.MaxFilter(3))) > 0.8

    # what leaves no text: noise, a blank image, half of the image taken for ink
    noise = np.random.default_rng(0).integers(0, 2, size=(64, 256), dtype=np.uint8) * 255
    assert abs(text_agreement(text, Image.fromarray(noise))) < 0.2
    assert text_agreement(text, Image.new('L', text.size, 0)) == 0.0
    band = np.asarray(text).copy()
    band[32:] = 255
    assert text_agreement(text, Image.fromarray(band)) < 0.6


def test_capture_simulation_never_returns_an_image_without_its_text():
    # a grid of lines one pixel thick, the kind of text that blur and noise wipe out
    canvas = np.zeros((64, 256), dtype=np.uint8)
    canvas[12:52:8, 30:226] = 255
    canvas[12:52, 30:226:14] = 255
    clean = clean_text_image(canvas, (128, 32))

    def fallbacks(**kwargs):
        simulation = CaptureSimulation((128, 32), **kwargs)
        return sum(np.array_equal(simulation(canvas, np.random.default_rng(seed)), clean) for seed in range(40))

    harsh = {'binarize_prob': 1.0, 'blur': (2.5, 3.0), 'noise': (0.15, 0.2), 'min_resolution': 0.3}
    # nearly every try loses the lines, and the clean rendering stands in for a result without them
    assert fallbacks(**harsh) >= 30
    # without the check the damaged images are returned
    assert fallbacks(min_agreement=-1.0, **harsh) == 0
    # the default settings leave even hairlines readable, so they are not replaced by clean images
    assert fallbacks() <= 2

    # if no try keeps the text, the clean rendering stands in
    impossible = CaptureSimulation((128, 32), min_agreement=1.1)
    np.testing.assert_array_equal(impossible(canvas, np.random.default_rng(0)), clean)


def test_bright_paper_is_not_taken_for_ink(fonts):
    from PIL import Image
    from font_reconstructor.dataset.capture import text_agreement

    # text under strongly uneven light, which would blow out the bright side of a photo
    simulation = CaptureSimulation((128, 32), lighting=0.6, noise=(0.0, 0.0), jpeg_prob=0.0, min_resolution=1.0)
    text = Image.fromarray(text_canvas(fonts))
    for seed in range(40):
        rng = np.random.default_rng(seed)
        photo = simulation._photograph(text, rng)
        # the lit paper stays a plane, it is not cut off at white
        assert (np.asarray(photo) >= 255).mean() < 0.01, seed
        assert text_agreement(text, simulation._clean_up(photo, rng)) > 0.95, seed


def test_stroke_changes_spare_hairlines():
    from PIL import Image

    simulation = CaptureSimulation((128, 32), stroke_change_prob=1.0, blur=(0.01, 0.01), motion_blur_prob=0.0)
    hairlines = np.zeros((64, 256), dtype=np.uint8)
    hairlines[10:54:6, 20:236] = 255   # lines one pixel thick with five pixels between them
    heavy = np.zeros((64, 256), dtype=np.uint8)
    heavy[16:48, 40:216] = 255

    changed_heavy = 0
    for seed in range(20):
        thin = np.asarray(simulation._print_and_focus(Image.fromarray(hairlines), np.random.default_rng(seed)))
        # neither wiped out nor thickened to three times their width
        assert 0.6 * hairlines.sum() <= thin.astype(np.float64).sum() <= 1.6 * hairlines.sum(), seed
        thick = np.asarray(simulation._print_and_focus(Image.fromarray(heavy), np.random.default_rng(seed)))
        changed_heavy += abs(float(thick.astype(np.float64).sum()) / heavy.sum() - 1.0) > 0.03
    assert changed_heavy == 20, 'strokes that can bear it are still thickened or thinned'


def test_variants_draw_a_font_as_another_style(fonts):
    from font_reconstructor.dataset import render_fingerprint, render_text

    ttf = fonts.ttf(1)
    plain = render_text(ttf, 'bad', (256, 64))
    outline = render_text(ttf, 'bad', (256, 64), variant=Variant(outline=0.05))
    heavy = render_text(ttf, 'bad', (256, 64), variant=Variant(weight=0.05))
    slanted = render_text(ttf, 'l', (64, 64), variant=Variant(slant=0.3))
    upright = render_text(ttf, 'l', (64, 64))

    def ink(image):
        return (image > 127).mean()

    # a heavier weight has more ink
    assert ink(heavy) > 1.3 * ink(plain)
    # an outline is hollow: where the filled stem of the b is, the outlined one is dark in the middle
    filled_columns = (plain > 127).sum(axis=0)
    stem = int(np.argmax(filled_columns))
    assert (outline[:, stem] > 127).sum() < 0.5 * filled_columns[stem]

    # a positive slant leans the top of a stem to the left of its foot
    def centre(image, rows):
        columns = np.where(image[rows] > 127)[1]
        return columns.mean()
    top, bottom = slice(8, 20), slice(44, 56)
    assert abs(centre(upright, top) - centre(upright, bottom)) < 1.5
    assert centre(slanted, top) < centre(slanted, bottom) - 3

    fingerprint = render_fingerprint(ttf, 'ab', (32, 32), 2, variant=Variant(outline=0.05))
    assert fingerprint.shape == (32, 32, 2)
    assert not np.array_equal(fingerprint, render_fingerprint(ttf, 'ab', (32, 32), 2))


def test_synthetic_variants_are_fonts_of_their_own(fonts_dir, tmp_path):
    root, annotation_file = fonts_dir
    counts = {'outline': 2, 'italic': 5}
    fonts = FontSet(str(root), str(annotation_file), synthetic_variants=counts)

    # three real fonts, two outlines and, since there are only three fonts to slant, three italics
    assert len(fonts) == 3 + 2 + 3
    assert [fonts.is_synthetic(index) for index in range(len(fonts))] == [False] * 3 + [True] * 5
    assert fonts.styles == ['regular'] * 3 + ['outline'] * 2 + ['italic'] * 3
    assert fonts.style_names == ['italic', 'outline', 'regular']
    assert len(set(fonts.names)) == len(fonts)
    for index in range(3, len(fonts)):
        base = fonts.files.index(fonts.files[index])
        # a variant is drawn from the file of its font and belongs to its family
        assert fonts.names[index].startswith(fonts.names[base])
        assert fonts.families[index] == fonts.families[base]
        assert fonts.charsets[index] == fonts.charsets[base]

    # the same choice every time, and part of what a cache depends on
    again = FontSet(str(root), str(annotation_file), synthetic_variants=counts)
    assert again.names == fonts.names
    assert again.signature() == fonts.signature()
    assert fonts.signature() != FontSet(str(root), str(annotation_file)).signature()

    with pytest.raises(ValueError, match='shadow'):
        FontSet(str(root), str(annotation_file), synthetic_variants={'shadow': 1})

    # the dataset draws a variant differently from its font, text and fingerprint alike
    dataset = make_dataset(fonts, tmp_path, total_samples=8, group_by_font=True, cache_images=False,
                           cache_fingerprints=False)
    variant = 3
    base = fonts.files.index(fonts.files[variant])
    assert not np.array_equal(dataset.generate_text_image(variant, 'bad'), dataset.generate_text_image(base, 'bad'))
    assert not np.array_equal(dataset.generate_font_fingerprint(variant), dataset.generate_font_fingerprint(base))


def test_synthetic_variants_are_not_validated_on(fonts_dir, tmp_path):
    root, annotation_file = fonts_dir
    fonts = FontSet(str(root), str(annotation_file), synthetic_variants={'outline': 3, 'italic': 3})
    train_loader, valid_loader = make_train_valid_loaders(
        fonts, random_seed=7, total_samples=90, validation_split=0.34, batch_size=8, num_workers=0,
        cache_dir=str(tmp_path / 'cache'),
    )
    train_fonts = set(train_loader.dataset.dataset.font_indices)
    valid_fonts = set(valid_loader.dataset.dataset.font_indices)
    assert valid_fonts and not any(fonts.is_synthetic(index) for index in valid_fonts)
    assert any(fonts.is_synthetic(index) for index in train_fonts)
    # the variants of a held out font are in its family, so they are not trained on
    held_out_families = {fonts.families[index] for index in valid_fonts}
    assert not any(fonts.families[index] in held_out_families for index in train_fonts)
