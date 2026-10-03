import copy

import pytest
import torch

from font_reconstructor.config import ConfigParser
from font_reconstructor.utils import mps_is_available

from conftest import load_script


def make_config(fonts_dir, tmp_path, device='cpu', epochs=2, corpus_file=None, **trainer):
    """
    a tiny run: three fonts, one of them held out for validation, batches of two fonts with four samples each
    """
    root, annotation_file = fonts_dir
    dataset = {
        'fonts_dir': str(root),
        'annotation_file': str(annotation_file),
        'font_size': 32,
        'text_length': [3, 8],
        'text_image_dims': [128, 32],
        'font_fingerprint_dims': [32, 32],
        'cache_dir': str(tmp_path / 'cache'),
    }
    if corpus_file is not None:
        dataset['corpus_files'] = [str(corpus_file)]
    return {
        'name': 'smoke',
        'seed': 1,
        'n_gpu': 0 if device == 'cpu' else 1,
        'device': device,
        'arch': {'type': 'CompactAutoEncoder', 'args': {'latent_dim': 16, 'base_channels': 4}},
        'dataset': dataset,
        'data_loader': {'args': {
            'random_seed': 42, 'total_samples': 48, 'validation_split': 0.34,
            'batch_fonts': 2, 'batch_samples_per_font': 4,
            'batch_size': 8, 'shuffle': True, 'num_workers': 0, 'pin_memory': True,
        }},
        'clustering_data_loader': {'args': {
            'random_seed': 123, 'samples_per_font': 4, 'batch_size': 8, 'num_workers': 0,
        }},
        'optimizer': {'type': 'Adam', 'args': {'lr': 0.001}},
        'loss': 'l1_loss',
        'contrastive_loss': {'weight': 0.5, 'temperature': 0.1},
        'metrics': ['mean_squared_error'],
        'lr_scheduler': {'type': 'StepLR', 'args': {'step_size': 50, 'gamma': 0.1}},
        'trainer': {
            'epochs': epochs, 'save_dir': str(tmp_path / 'saved'), 'save_period': 1, 'verbosity': 2,
            'monitor': 'min val_loss', 'early_stop': 10, 'topk': [1, 2], 'tensorboard': False, **trainer,
        },
    }


def run_dir(tmp_path, run_id):
    return tmp_path / 'saved' / 'models' / 'smoke' / run_id


def test_train_resume_and_test(fonts_dir, corpus_file, tmp_path):
    train = load_script('train')
    test = load_script('test')
    config = make_config(fonts_dir, tmp_path, epochs=3, corpus_file=corpus_file, keep_last_checkpoints=2)

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

    config['style_head'] = {'weight': 0.2}
    config['trainer']['epochs'] = 2
    config['trainer']['figure_period'] = 2

    train.main(ConfigParser(config, run_id='tensorboard'))
    log_dir = tmp_path / 'saved' / 'log' / 'smoke' / 'tensorboard'
    assert list(log_dir.glob('events.out.tfevents.*'))

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(log_dir), size_guidance={'images': 0})
    events.Reload()
    tags = events.Tags()
    assert {'samples', 'worst_cases', 'glyph_error', 'latent_space', 'latent_comparison', 'style_predictions',
            'identification/by_rank', 'identification/by_style', 'identification/by_text_length'} <= set(tags['images'])
    # figures are written every second epoch here, scalars every epoch
    assert [event.step for event in events.Images('samples')] == [2]
    assert {'mrr/val', 'recon_skill/val', 'style_acc/val', 'style_acc/train', 'loss/val'} <= set(tags['scalars'])
    assert [event.step for event in events.Scalars('mrr/val')] == [1, 2]


def test_train_with_conditioned_decoder_and_adversarial_loss(fonts_dir, tmp_path):
    train = load_script('train')
    config = make_config(fonts_dir, tmp_path, epochs=2)
    config['arch']['args'].update({'decoder_type': 'conditioned', 'decoder_channels': 8, 'decoder_blocks': 2})
    config['adversarial_loss'] = {'weight': 0.05, 'discriminator_channels': 8, 'start_epoch': 2}
    config['data_loader']['args']['validation_augmentations'] = True
    config['clustering_data_loader']['args']['random_augmentations'] = True
    config['dataset']['capture'] = {'binarize_prob': 1.0}

    train.main(ConfigParser(copy.deepcopy(config), run_id='adversarial'))
    checkpoint_path = run_dir(tmp_path, 'adversarial') / 'checkpoint-epoch2.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    # the discriminator is stored with the checkpoint, so that a resumed run continues the same game
    assert set(checkpoint['extras']) == {'discriminator', 'discriminator_optimizer', 'lr_scheduler'}

    resumed = copy.deepcopy(config)
    resumed['trainer']['epochs'] = 3
    train.main(ConfigParser(resumed, resume=checkpoint_path, run_id='adversarial_resumed'))
    assert (run_dir(tmp_path, 'adversarial_resumed') / 'checkpoint-epoch3.pth').exists()


def test_train_with_style_head(fonts_dir, tmp_path):
    train = load_script('train')
    test = load_script('test')
    root, _ = fonts_dir
    annotation_file = tmp_path / 'fonts_with_styles.csv'
    annotation_file.write_text(
        'font,file,supported_charset,style\n'
        'Sans,DejaVuSans.ttf,a b c d e f g h,upright\n'
        'Serif,DejaVuSerif.ttf,a b c d e f g h,serif\n'
        'Mono,DejaVuSansMono.ttf,a b c d e f g h,upright\n'
    )
    config = make_config(fonts_dir, tmp_path, epochs=2)
    config['dataset']['annotation_file'] = str(annotation_file)
    config['style_head'] = {'weight': 0.2}

    train.main(ConfigParser(copy.deepcopy(config), run_id='style'))
    checkpoint_path = run_dir(tmp_path, 'style') / 'checkpoint-epoch2.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    # one score per style
    assert checkpoint['state_dict']['style_head.weight'].shape == (2, 16)

    log = test.main(ConfigParser(copy.deepcopy(config), resume=checkpoint_path, run_id='style_test'))
    assert {'style_acc', 'style_balanced_acc'} <= set(log)
    assert 0 <= log['style_acc'] <= 1

    # the head can also be trained as a probe that leaves the encoder alone
    config['style_head'] = {'weight': 1.0, 'detach': True}
    train.main(ConfigParser(copy.deepcopy(config), run_id='style_probe'))


def test_one_cycle_schedule_is_stepped_every_batch_and_resumed(fonts_dir, tmp_path):
    train = load_script('train')
    config = make_config(fonts_dir, tmp_path, epochs=4)
    config['lr_scheduler'] = {'type': 'OneCycleLR', 'args': {'max_lr': 0.01, 'pct_start': 0.25, 'div_factor': 10}}
    steps_per_epoch = 4  # 32 training samples in batches of 8

    # stop after two of the four epochs, as an interrupted run would
    first = copy.deepcopy(config)
    first['trainer']['early_stop'] = 1
    first['trainer']['monitor'] = 'min epoch'
    train.main(ConfigParser(first, run_id='one_cycle'))
    checkpoint_path = sorted(run_dir(tmp_path, 'one_cycle').glob('checkpoint-*.pth'))[-1]
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    scheduler = checkpoint['extras']['lr_scheduler']
    # the schedule spans all epochs of the run, and has been stepped once per batch
    assert scheduler['total_steps'] == 4 * steps_per_epoch
    assert scheduler['last_epoch'] == checkpoint['epoch'] * steps_per_epoch
    assert checkpoint['epoch'] < 4

    # a resumed run continues the schedule to its end instead of starting it again
    train.main(ConfigParser(copy.deepcopy(config), resume=checkpoint_path, run_id='one_cycle_resumed'))
    final = torch.load(run_dir(tmp_path, 'one_cycle_resumed') / 'checkpoint-epoch4.pth', map_location='cpu',
                       weights_only=True)
    assert final['extras']['lr_scheduler']['last_epoch'] == 4 * steps_per_epoch
    assert final['optimizer']['param_groups'][0]['lr'] < 0.01 / 10


def test_lr_scheduler_intervals(fonts_dir, tmp_path):
    from font_reconstructor import factory

    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    config = make_config(fonts_dir, tmp_path, epochs=5)

    # the template default: stepped once per epoch
    scheduler, interval = factory.build_lr_scheduler(config, optimizer, steps_per_epoch=7)
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR) and interval == 'epoch'

    # the length of a schedule comes from the run where the config leaves it out
    config['lr_scheduler'] = {'type': 'OneCycleLR', 'args': {'max_lr': 0.1}}
    scheduler, interval = factory.build_lr_scheduler(config, optimizer, steps_per_epoch=7)
    assert interval == 'batch' and scheduler.total_steps == 5 * 7
    config['lr_scheduler'] = {'type': 'CosineAnnealingLR', 'interval': 'batch', 'args': {}}
    scheduler, interval = factory.build_lr_scheduler(config, optimizer, steps_per_epoch=7)
    assert interval == 'batch' and scheduler.T_max == 5 * 7
    config['lr_scheduler'] = {'type': 'CosineAnnealingLR', 'args': {}}
    assert factory.build_lr_scheduler(config, optimizer, steps_per_epoch=7)[0].T_max == 5

    with pytest.raises(ValueError):
        config['lr_scheduler'] = {'type': 'OneCycleLR', 'interval': 'epoch', 'args': {'max_lr': 0.1}}
        factory.build_lr_scheduler(config, optimizer, steps_per_epoch=7)
    del config['lr_scheduler']
    assert factory.build_lr_scheduler(config, optimizer, steps_per_epoch=7) == (None, 'epoch')


def test_lr_range_test_readings():
    import numpy as np
    find_lr = load_script('scripts/find_lr')

    # the moving average is corrected for its start, so a constant stays a constant
    assert np.allclose(find_lr.smooth([2.0] * 5), 2.0)
    assert find_lr.smooth([0.0, 10.0])[-1] == pytest.approx((0.9 * 0.0 + 0.1 * 10.0 + 0.0) / (1 - 0.9 ** 2))

    # a loss that falls fastest at 1e-3, is lowest at 1e-2 and rises after it
    rates = np.logspace(-5, 0, 101)
    log_rate = np.log10(rates)
    loss = np.where(log_rate < -2, 1 - 1 / (1 + np.exp(-4 * (log_rate + 3))), (log_rate + 2) ** 2 * 0.5 + 0.018)
    marks = find_lr.read_marks(rates, loss)
    assert marks['steepest'] == pytest.approx(1e-3, rel=0.3)
    assert marks['minimum'] == pytest.approx(1e-2, rel=0.3)


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
    # the output channels may still be written in the config, as long as they agree with the fonts
    config['arch']['args']['decoder_output_channels'] = 8
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
