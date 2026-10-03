import collections
import pickle

import numpy as np
import pytest
import torch
from PIL import features

from font_reconstructor.dataset import (CaptureSimulation, FontBalancedBatchSampler, FontSet, RandomTextImageDataset,
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
    assert [indices.tolist() for indices in grouped.sample_indices_by_font()] == [[0, 2, 4, 6], [1, 3, 5]]

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
