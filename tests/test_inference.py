import io

import matplotlib
import numpy as np
import pytest
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from font_reconstructor import preprocessing
from font_reconstructor.dataset.capture import _perspective_coefficients
from font_reconstructor.index import FontIndex
from font_reconstructor.inference import FontIdentifier, export_model, load_model, model_id
from font_reconstructor.model import CompactAutoEncoder

from conftest import load_script

FONT_DIR = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf'
LINES = ['bad cab faded head', 'a deaf bee had a bag', 'feed each caged beach']


def page_image(lines=LINES, size=(640, 300), font_file='DejaVuSans.ttf', font_size=34, paper=235, ink=20):
    """
    a page of dark text on light paper, as a scan would show it
    """
    image = Image.new('L', size, paper)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(FONT_DIR / font_file), font_size)
    for number, line in enumerate(lines):
        draw.text((40, 30 + number * 80), line, fill=ink, font=font)
    return image


def photo_of(page, size=(900, 700), corners=((130, 90), (780, 150), (740, 560), (90, 480)), surround=35):
    """
    the page as a photo: seen at an angle, with something darker around it
    """
    width, height = page.size
    coefficients = _perspective_coefficients(corners, [(0, 0), (width, 0), (width, height), (0, height)])
    mask = Image.new('L', page.size, 255).transform(size, Image.PERSPECTIVE, coefficients, Image.BILINEAR)
    warped = page.transform(size, Image.PERSPECTIVE, coefficients, Image.BICUBIC)
    photo = Image.new('L', size, surround)
    photo.paste(warped, (0, 0), mask)
    return photo


def test_scan_is_cleaned_and_cut_into_lines_and_pieces():
    page = page_image()
    pixels = preprocessing.load_grey(page)
    mask, corners = preprocessing.find_page(pixels)
    # a scan is all page
    assert mask.all() and corners is None

    ink = preprocessing.clean(pixels)
    assert ink.dtype == np.float32 and ink.min() == 0.0 and ink.max() == 1.0
    # the paper is black and the strokes are white
    assert np.median(ink) == 0.0 and 0.02 < (ink > 0.5).mean() < 0.3

    lines = preprocessing.find_lines(ink)
    assert len(lines) == 3
    assert all(25 < bottom - top < 60 for top, bottom in lines)

    pieces = preprocessing.prepare(page)
    assert [piece.line for piece in pieces] == sorted(piece.line for piece in pieces)
    assert {piece.line for piece in pieces} == {0, 1, 2}
    for piece in pieces:
        assert piece.image.shape == (64, 256) and piece.image.dtype == np.uint8
        assert piece.image.max() > 200 and np.median(piece.image) == 0
        left, top, right, bottom = piece.box
        # no piece is much wider than the texts the model was trained on
        assert (right - left) / (bottom - top) < 10

    # bright text on a dark page gives the same pieces
    inverted = preprocessing.prepare(Image.eval(page, lambda value: 255 - value))
    assert len(inverted) == len(pieces)
    assert np.abs(inverted[0].image.astype(int) - pieces[0].image.astype(int)).mean() < 12

    assert preprocessing.prepare(Image.new('L', (300, 200), 240)) == []


def test_photo_of_a_page_is_straightened():
    page = page_image()
    corners = ((130, 90), (780, 150), (740, 560), (90, 480))
    photo = photo_of(page, corners=corners)

    mask, found = preprocessing.find_page(preprocessing.load_grey(photo))
    assert 0.3 < mask.mean() < 0.7
    # the corners are found to a few pixels
    assert np.abs(np.array(found) - np.array(corners)).max() < 8

    ink = preprocessing.clean_page(photo)
    lines = preprocessing.find_lines(ink)
    # the lines are level again: three of them, as high as on the page, and nothing of the edge of the page
    assert len(lines) == 3
    heights = [bottom - top for top, bottom in lines]
    assert max(heights) < 1.5 * min(heights)
    assert ink[:, :3].max() == 0 and ink[:3].max() == 0

    straight = preprocessing.prepare(photo)
    assert {piece.line for piece in straight} == {0, 1, 2}
    # without straightening the lines run into each other
    tilted = preprocessing.clean(preprocessing.load_grey(photo), mask)
    assert len(preprocessing.find_lines(tilted)) != 3 or max(
        bottom - top for top, bottom in preprocessing.find_lines(tilted)) > 1.5 * max(heights)


def test_skew_is_measured_and_removed():
    page = page_image()
    pixels = preprocessing.load_grey(page.rotate(6, Image.BICUBIC, expand=True, fillcolor=235))
    ink = preprocessing.clean(pixels)
    angle = preprocessing.estimate_skew(ink)
    assert angle == pytest.approx(6, abs=0.75)
    assert len(preprocessing.find_lines(preprocessing.deskew(ink, angle))) == 3
    assert preprocessing.estimate_skew(preprocessing.clean(preprocessing.load_grey(page))) == pytest.approx(0, abs=0.5)


def test_font_index(tmp_path):
    path = tmp_path / 'fonts.sqlite'
    index = FontIndex(path, dim=3, model_id='model-a')
    assert len(index) == 0 and index.search([1.0, 0.0, 0.0]) == []

    index.add('East', [[2.0, 0.1, 0.0], [4.0, -0.1, 0.0]], family='Compass', style='regular', source='test')
    index.add('North', [[0.0, 3.0, 0.0]], family='Compass', style='bold')
    index.add('Up', [[0.0, 0.0, 1.0]])
    assert len(index) == 3 and 'East' in index and 'West' not in index
    assert [font['name'] for font in index.fonts()] == ['East', 'North', 'Up']
    assert index.fonts()[0]['samples'] == 2 and index.fonts()[0]['source'] == 'test'
    assert index.vectors('East') == pytest.approx(np.array([[2.0, 0.1, 0.0], [4.0, -0.1, 0.0]]), abs=1e-3)

    # the length of a query does not matter, only its direction
    matches = index.search([10.0, 1.0, 0.0], k=2)
    assert [match['name'] for match in matches] == ['East', 'North']
    assert [match['rank'] for match in matches] == [1, 2]
    assert matches[0]['similarity'] == pytest.approx(10 / np.sqrt(101), abs=1e-3) and matches[0]['family'] == 'Compass'
    # several vectors of one font are averaged after scaling them to unit length
    assert index.search([[100.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], k=1)[0]['name'] == 'North'

    with pytest.raises(ValueError, match='already'):
        index.add('East', [[1.0, 0.0, 0.0]])
    index.add('East', [[0.0, 0.0, 5.0]], extend=True)
    assert index.fonts()[0]['samples'] == 3 and index.fonts()[0]['family'] == 'Compass'
    index.add('East', [[0.0, 0.0, 5.0]], replace=True)
    assert index.search([0.0, 0.0, 1.0], k=1)[0]['similarity'] == pytest.approx(1.0, abs=1e-4)
    assert index.remove('Up') and not index.remove('Up')
    with pytest.raises(ValueError):
        index.add('Wide', [[1.0, 2.0, 3.0, 4.0]])
    with pytest.raises(ValueError):
        index.add('Broken', [[np.nan, 0.0, 0.0]])
    index.close()

    # the file holds everything
    again = FontIndex(path)
    assert again.dim == 3 and [font['name'] for font in again.fonts()] == ['East', 'North']
    again.close()
    with pytest.raises(ValueError, match='another model'):
        FontIndex(path, dim=3, model_id='model-b')
    with pytest.raises(ValueError, match='size'):
        FontIndex(path, dim=8)
    with pytest.raises(ValueError, match='new font index'):
        FontIndex(tmp_path / 'other.sqlite')


@pytest.fixture()
def exported_model(fonts, corpus_file, tmp_path):
    """
    a small untrained model, exported the way a trained one is
    """
    torch.manual_seed(0)
    arch = {'base_channels': 4, 'latent_dim': 16, 'decoder_channels': 4}
    model = CompactAutoEncoder(decoder_output_channels=fonts.num_glyphs, style_classes=len(fonts.style_names), **arch)
    config = {
        'arch': {'type': 'CompactAutoEncoder', 'args': arch},
        'dataset': {'font_size': 32, 'layout_engine': 'basic', 'text_length': [3, 8], 'text_image_dims': [128, 32],
                    'font_fingerprint_dims': [32, 32], 'render_scale': 2, 'corpus_files': [str(corpus_file)]},
    }
    checkpoint = tmp_path / 'checkpoint.pth'
    torch.save({'state_dict': model.state_dict(), 'config': config, 'epoch': 3}, checkpoint)
    path = tmp_path / 'exported' / 'model.pth'
    exported = export_model(checkpoint, path, fonts, notes={'top1': 0.5})
    return path, exported, model


def test_exported_model_needs_no_training_files(exported_model):
    path, exported, model = exported_model
    assert exported['glyphs'] == 'abcdefgh' and exported['epoch'] == 3 and exported['notes'] == {'top1': 0.5}
    assert exported['arch']['args']['input_dims'] == [32, 128] and exported['arch']['args']['style_classes'] == 1
    assert exported['model_id'] == model_id(model.state_dict()) and len(exported['model_id']) == 16

    loaded, info = load_model(path)
    assert not loaded.training and 'state_dict' not in info
    image = torch.rand(2, 1, 32, 128)
    assert torch.allclose(loaded.encode(image), model.eval().encode(image))
    # other weights are another model
    other = CompactAutoEncoder(decoder_output_channels=8, base_channels=4, latent_dim=16, decoder_channels=4)
    assert model_id(other.state_dict()) != exported['model_id']

    broken = path.with_name('broken.pth')
    torch.save({'format': 99}, broken)
    with pytest.raises(ValueError, match='format'):
        load_model(broken)


def test_identifier_adds_fonts_and_finds_them(exported_model, tmp_path):
    path, exported, _ = exported_model
    identifier = FontIdentifier(path, tmp_path / 'fonts.sqlite', device='cpu')
    assert identifier.index.meta('model_id') == exported['model_id'] and identifier.dims == (128, 32)

    described = identifier.describe_font(FONT_DIR / 'DejaVuSans-Bold.ttf')
    assert described['name'] == 'DejaVu Sans Bold' and described['style'] == 'bold' and described['charset'] == 'abcdefgh'

    # sample images are made like the training images, and the same every time
    images = identifier.render_font(FONT_DIR / 'DejaVuSans.ttf', samples=6, seed=1)
    assert len(images) == 6 and images[0].shape == (32, 128) and images[0].dtype == np.uint8
    again = identifier.render_font(FONT_DIR / 'DejaVuSans.ttf', samples=6, seed=1)
    assert all(np.array_equal(first, second) for first, second in zip(images, again))
    assert not np.array_equal(images[0], identifier.render_font(FONT_DIR / 'DejaVuSans.ttf', samples=1, seed=2)[0])
    assert identifier.encode(images).shape == (6, 16)

    for file in ('DejaVuSans.ttf', 'DejaVuSerif.ttf', 'DejaVuSansMono.ttf'):
        added = identifier.add_font(FONT_DIR / file, samples=8)
        assert added['samples'] == 8
    names = [font['name'] for font in identifier.index.fonts()]
    assert names == ['DejaVu Sans Book', 'DejaVu Serif Book', 'DejaVu Sans Mono Book']
    assert identifier.index.fonts()[0]['source'] == 'added by user'
    with pytest.raises(ValueError, match='already'):
        identifier.add_font(FONT_DIR / 'DejaVuSans.ttf', samples=2)
    identifier.add_font(FONT_DIR / 'DejaVuSans.ttf', name='My Sans', samples=2)
    assert 'My Sans' in identifier.index
    # a font is closest to its own samples
    for font in identifier.index.fonts():
        assert identifier.index.search(identifier.index.vectors(font['name']), k=1)[0]['name'] == font['name']

    result = identifier.identify(page_image(font_file='DejaVuSerif.ttf'), k=3)
    assert len(result.pieces) >= 3 and len(result.matches) == 3
    assert [match['similarity'] for match in result.matches] == sorted(
        (match['similarity'] for match in result.matches), reverse=True)
    assert [line['line'] for line in result.lines] == [0, 1, 2]
    assert sum(line['pieces'] for line in result.lines) == len(result.pieces)
    assert result.style[0] == 'regular' and result.style[1] == pytest.approx(1.0)

    empty = identifier.identify(Image.new('L', (200, 100), 255))
    assert empty.matches == [] and empty.pieces == []

    # an index of another model is refused
    other = FontIndex(tmp_path / 'other.sqlite', dim=16, model_id='0123456789abcdef')
    other.close()
    with pytest.raises(ValueError, match='another model'):
        FontIdentifier(path, tmp_path / 'other.sqlite', device='cpu')


def test_identify_script(exported_model, tmp_path, capsys):
    identify = load_script('identify')
    path, _, _ = exported_model
    common = ['--model', str(path), '--index', str(tmp_path / 'fonts.sqlite'), '--device', 'cpu']

    assert identify.main(common + ['add', str(FONT_DIR / 'DejaVuSans.ttf'), str(FONT_DIR / 'DejaVuSerif.ttf'),
                                   '--samples', '6']) == 0
    assert "added 'DejaVu Serif Book'" in capsys.readouterr().out
    assert identify.main(common + ['list']) == 0
    listing = capsys.readouterr().out
    assert 'DejaVu Sans Book' in listing and '2 fonts' in listing

    photo = tmp_path / 'photo.png'
    photo_of(page_image()).save(photo)
    assert identify.main(common + ['query', str(photo), '-k', '2', '--debug-dir', str(tmp_path / 'pieces')]) == 0
    output = capsys.readouterr().out
    assert 'in 3 lines' in output and 'all text together' in output and 'line by line' in output
    assert len(list((tmp_path / 'pieces').glob('piece_*.png'))) >= 3
    assert identify.main(common + ['query', str(photo), '--json']) == 0
    assert '"matches"' in capsys.readouterr().out

    blank = tmp_path / 'blank.png'
    Image.new('L', (200, 100), 255).save(blank)
    assert identify.main(common + ['query', str(blank)]) == 1
    assert 'No text was found' in capsys.readouterr().out

    assert identify.main(common + ['remove', 'DejaVu Sans Book']) == 0
    assert identify.main(common + ['remove', 'DejaVu Sans Book']) == 1
    with pytest.raises(SystemExit):
        identify.main(['--model', str(tmp_path / 'missing.pth'), 'list'])


def test_image_with_transparency_is_read_on_a_white_page():
    page = page_image().convert('RGBA')
    pixels = np.array(page)
    pixels[..., 3] = np.where(pixels[..., 0] > 128, 0, 255)   # the paper is see-through
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format='PNG')
    buffer.seek(0)
    assert len(preprocessing.find_lines(preprocessing.clean(preprocessing.load_grey(Image.open(buffer))))) == 3
