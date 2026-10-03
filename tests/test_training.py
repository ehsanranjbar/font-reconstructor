import copy

import pytest
import torch

from font_reconstructor.config import ConfigParser
from font_reconstructor.utils import mps_is_available

from conftest import load_script


def make_config(fonts_dir, tmp_path, device='cpu', epochs=2, **trainer):
    root, annotation_file = fonts_dir
    return {
        'name': 'smoke',
        'seed': 1,
        'n_gpu': 0 if device == 'cpu' else 1,
        'device': device,
        'arch': {'type': 'AutoEncoder', 'args': {'latent_dim': 16, 'base_conv_filters': 4}},
        'dataset': {
            'fonts_dir': str(root),
            'annotation_file': str(annotation_file),
            'font_size': 32,
            'text_length': [3, 8],
            'text_image_dims': [128, 32],
            'font_fingerprint_dims': [32, 32],
            'cache_dir': str(tmp_path / 'cache'),
        },
        'data_loader': {'args': {
            'random_seed': 42, 'total_samples': 48, 'validation_split': 0.25,
            'batch_size': 8, 'shuffle': True, 'num_workers': 0, 'pin_memory': True,
        }},
        'clustering_data_loader': {'args': {
            'random_seed': 123, 'samples_per_font': 4, 'batch_size': 8, 'num_workers': 0,
        }},
        'optimizer': {'type': 'Adam', 'args': {'lr': 0.001}},
        'loss': 'mse_loss',
        'metrics': ['mean_squared_error'],
        'lr_scheduler': {'type': 'StepLR', 'args': {'step_size': 50, 'gamma': 0.1}},
        'trainer': {
            'epochs': epochs, 'save_dir': str(tmp_path / 'saved'), 'save_period': 1, 'verbosity': 2,
            'monitor': 'min val_loss', 'early_stop': 10, 'topk': [1, 2], 'tensorboard': False, **trainer,
        },
    }


def run_dir(tmp_path, run_id):
    return tmp_path / 'saved' / 'models' / 'smoke' / run_id


def test_train_resume_and_test(fonts_dir, tmp_path):
    train = load_script('train')
    test = load_script('test')
    config = make_config(fonts_dir, tmp_path, epochs=3, keep_last_checkpoints=2)

    train.main(ConfigParser(copy.deepcopy(config), run_id='first'))
    saved = sorted(path.name for path in run_dir(tmp_path, 'first').glob('*.pth'))
    assert saved == ['checkpoint-epoch2.pth', 'checkpoint-epoch3.pth', 'model_best.pth']

    # checkpoints hold plain data only and load without the classes of this project
    checkpoint_path = run_dir(tmp_path, 'first') / 'checkpoint-epoch3.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    assert checkpoint['epoch'] == 3
    assert type(checkpoint['config']) is dict
    assert not any(key.startswith('module.') for key in checkpoint['state_dict'])

    # resuming continues with the next epoch
    resumed = copy.deepcopy(config)
    resumed['trainer']['epochs'] = 4
    train.main(ConfigParser(resumed, resume=checkpoint_path, run_id='second'))
    assert [path.name for path in run_dir(tmp_path, 'second').glob('checkpoint-*.pth')] == ['checkpoint-epoch4.pth']

    log = test.main(ConfigParser(copy.deepcopy(config), resume=checkpoint_path, run_id='test'))
    assert set(log) == {'loss', 'top1_acc', 'top2_acc', 'mean_squared_error'}
    assert log['loss'] > 0
    assert 0 <= log['top1_acc'] <= log['top2_acc'] <= 1


def test_train_with_tensorboard(fonts_dir, tmp_path):
    train = load_script('train')
    config = make_config(fonts_dir, tmp_path, epochs=1, tensorboard=True)

    train.main(ConfigParser(config, run_id='tensorboard'))
    assert list((tmp_path / 'saved' / 'log' / 'smoke' / 'tensorboard').glob('events.out.tfevents.*'))


def test_train_without_validation(fonts_dir, tmp_path):
    train = load_script('train')
    config = make_config(fonts_dir, tmp_path, epochs=1, monitor='off')
    config['data_loader']['args']['validation_split'] = 0.0
    config['lr_scheduler'] = {'type': 'ReduceLROnPlateau', 'args': {}}
    del config['clustering_data_loader']

    train.main(ConfigParser(config, run_id='no_validation'))
    assert (run_dir(tmp_path, 'no_validation') / 'checkpoint-epoch1.pth').exists()


def test_legacy_config_layout(fonts_dir, tmp_path):
    train = load_script('train')
    config = make_config(fonts_dir, tmp_path, epochs=1)
    dataset = config.pop('dataset')
    config['arch'] = {'type': 'AE2', 'args': {'latent_dim': 16, 'base_conv_filters': 4, 'decoder_output_channels': 8}}
    config['data_loader'] = {'type': 'RTIDataLoader', 'args': {**dataset, **config['data_loader']['args']}}
    config['clustering_data_loader'] = {'type': 'RTIDataLoader', 'args': {
        **dataset, 'random_seed': 123, 'total_samples': 12, 'group_by_font': True, 'random_augmentions': False,
        'batch_size': 8, 'shuffle': False, 'validation_split': 0.0, 'num_workers': 0,
    }}

    with pytest.warns(DeprecationWarning, match='random_augmentions'):
        train.main(ConfigParser(config, run_id='legacy'))
    assert (run_dir(tmp_path, 'legacy') / 'model_best.pth').exists()


def test_model_has_to_match_the_dataset(fonts_dir, tmp_path):
    train = load_script('train')
    config = make_config(fonts_dir, tmp_path, epochs=1)
    config['arch']['args']['decoder_output_channels'] = 42

    with pytest.raises(ValueError, match='decoder_output_channels'):
        train.main(ConfigParser(config, run_id='mismatch'))


@pytest.mark.skipif(not mps_is_available(), reason='needs the MPS backend')
def test_train_and_test_on_mps(fonts_dir, tmp_path):
    train = load_script('train')
    test = load_script('test')
    config = make_config(fonts_dir, tmp_path, device='mps', epochs=1, tensorboard=True)

    train.main(ConfigParser(copy.deepcopy(config), run_id='mps'))
    checkpoint_path = run_dir(tmp_path, 'mps') / 'model_best.pth'
    assert checkpoint_path.exists()

    log = test.main(ConfigParser(copy.deepcopy(config), resume=checkpoint_path, run_id='mps_test'))
    assert log['loss'] > 0
