import importlib.util
import shutil
from pathlib import Path

import matplotlib
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# fonts that ship with matplotlib, so the tests need no font files of their own
FONT_FILES = ['DejaVuSans.ttf', 'DejaVuSerif.ttf', 'DejaVuSansMono.ttf']
CHARSET = 'a b c d e f g h'


@pytest.fixture(scope='session')
def fonts_dir(tmp_path_factory):
    """
    a fonts directory and its annotation file. One font declares a shorter charset, padded with null characters.
    """
    source = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf'
    root = tmp_path_factory.mktemp('fonts')
    for file in FONT_FILES:
        shutil.copy(source / file, root / file)

    rows = ['font,file,supported_charset']
    for i, file in enumerate(FONT_FILES):
        charset = CHARSET if i else CHARSET[:-3] + '\0 \0'
        rows.append(f'{Path(file).stem},{file},{charset}')
    annotation_file = root / 'fonts.csv'
    annotation_file.write_text('\n'.join(rows) + '\n')
    return root, annotation_file


@pytest.fixture()
def fonts(fonts_dir):
    from font_reconstructor.dataset import FontSet

    root, annotation_file = fonts_dir
    return FontSet(str(root), str(annotation_file), font_size=32)


def load_script(name):
    """
    import one of the scripts at the repository root as a module
    """
    spec = importlib.util.spec_from_file_location(f'{name}_script', REPO_ROOT / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
