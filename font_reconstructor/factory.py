"""
Builds the objects of a run from its configuration. This is the only module that knows the layout of config.json.
"""
import warnings
from functools import partial

import torch

import font_reconstructor.model.loss as module_loss
import font_reconstructor.model.metric as module_metric
import font_reconstructor.model.model as module_arch
from font_reconstructor.dataset import FontSet, TextCorpus, make_clustering_loader, make_train_valid_loaders

_FONTSET_KEYS = ('fonts_dir', 'annotation_file', 'font_size', 'layout_engine', 'synthetic_variants')
_RENDER_KEYS = ('text_length', 'text_image_dims', 'font_fingerprint_dims', 'number_ratio', 'render_scale', 'cache_dir')
_DATASET_KEYS = _FONTSET_KEYS + _RENDER_KEYS + ('corpus_files', 'capture')
# keys of the clustering loader in configs written before the shared `dataset` block, they have no effect anymore
_LEGACY_CLUSTERING_KEYS = ('group_by_font', 'shuffle', 'validation_split')


def dataset_config(config):
    """
    The settings shared by all datasets of a run.

    They live in the `dataset` block. Configs written before that block existed repeat them in the arguments
    of each data loader, those of the training loader are used then.
    """
    cfg = dict(config.get('dataset') or {})
    legacy_args = config['data_loader']['args']
    for key in _DATASET_KEYS:
        if key not in cfg and key in legacy_args:
            cfg[key] = legacy_args[key]
    return cfg


def loader_args(config, name):
    """
    arguments of the data loader `name` without the shared dataset settings
    """
    args = {key: value for key, value in config[name]['args'].items() if key not in _DATASET_KEYS}
    if 'random_augmentions' in args:
        warnings.warn("The config key 'random_augmentions' is deprecated, use 'random_augmentations'.",
                      DeprecationWarning, stacklevel=2)
        args.setdefault('random_augmentations', args['random_augmentions'])
        del args['random_augmentions']
    return args


def build_fontset(config):
    cfg = dataset_config(config)
    return FontSet(**{key: cfg[key] for key in _FONTSET_KEYS if key in cfg})


def build_corpus(config):
    """
    :return: the text corpus of the `corpus_files` dataset setting, or None to render random characters
    """
    files = dataset_config(config).get('corpus_files')
    return TextCorpus(files) if files else None


def build_train_valid_loaders(config, fonts, device=None, corpus=None, **overrides):
    """
    :param device: device the batches are moved to. Memory is only pinned for CUDA.
    :param corpus: text corpus of `build_corpus`
    :param overrides: loader arguments that replace those of the config
    """
    cfg = dataset_config(config)
    args = loader_args(config, 'data_loader')
    args.update({key: cfg[key] for key in _RENDER_KEYS if key in cfg})
    args['capture_options'] = cfg.get('capture')
    args.update(overrides)
    args['pin_memory'] = _pin_memory(args.get('pin_memory', False), device)
    return make_train_valid_loaders(fonts, corpus=corpus, **args)


def build_clustering_loader(config, fonts, device=None, corpus=None):
    """
    :param corpus: text corpus of `build_corpus`
    :return: the clustering loader, or None if the config has no `clustering_data_loader`
    """
    if not config.get('clustering_data_loader'):
        return None

    cfg = dataset_config(config)
    args = loader_args(config, 'clustering_data_loader')
    for key in _LEGACY_CLUSTERING_KEYS:
        args.pop(key, None)
    if 'total_samples' in args:
        # configs written before `samples_per_font` give the total over all fonts
        total_samples = args.pop('total_samples')
        args.setdefault('samples_per_font', max(1, total_samples // len(fonts)))
    args.update({key: cfg[key] for key in _RENDER_KEYS if key in cfg and key != 'font_fingerprint_dims'})
    args['capture_options'] = cfg.get('capture')
    args['pin_memory'] = _pin_memory(args.get('pin_memory', False), device)
    return make_clustering_loader(fonts, corpus=corpus, **args)


def build_model(config, fonts):
    """
    Build the model with its input and output shapes derived from the dataset settings and the font set.

    The shapes can still be written in the `arch` arguments, but they have to agree with the dataset.
    """
    cfg = dataset_config(config)
    text_width, text_height = cfg.get('text_image_dims', (128, 32))
    glyph_width, glyph_height = cfg.get('font_fingerprint_dims', (32, 32))
    derived = {
        'decoder_output_channels': fonts.num_glyphs,
        'input_dims': (text_height, text_width),
        'output_dims': (glyph_height, glyph_width),
    }
    if build_style_head(config):
        derived['style_classes'] = len(fonts.style_names)

    configured = config['arch']['args']
    kwargs = {}
    for key, value in derived.items():
        if key not in configured:
            kwargs[key] = value
        elif _as_tuple(configured[key]) != _as_tuple(value):
            raise ValueError(
                f"arch argument {key}={configured[key]} does not match the dataset, which needs {value}. "
                f"Remove it from the config to derive it from the dataset."
            )

    return config.init_obj('arch', module_arch, **kwargs)


def check_model_shapes(model, dataset):
    """
    run one sample of the dataset through the model and fail early if its output does not match the target
    """
    sample = dataset[0]
    image, target = sample['image'], sample['target']
    was_training = model.training
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0))
    model.train(was_training)

    if output.shape[1:] != target.shape:
        raise ValueError(
            f"The model maps an image of shape {tuple(image.shape)} to {tuple(output.shape[1:])}, "
            f"but the target fingerprint has shape {tuple(target.shape)}."
        )


def build_criterion(config):
    return getattr(module_loss, config['loss'])


def build_contrastive(config):
    """
    The contrastive loss on the latent vector, configured by the `contrastive_loss` block.

    :return: (loss function of (latent, font_index), weight), or (None, 0.0) if it is not configured
    """
    cfg = config.get('contrastive_loss') or {}
    weight = float(cfg.get('weight', 0.0))
    if not weight:
        return None, 0.0
    criterion = partial(module_loss.supervised_contrastive_loss, temperature=cfg.get('temperature', 0.1))
    return criterion, weight


# schedulers that plan every step in advance, so they are stepped after every batch
_BATCH_SCHEDULERS = ('OneCycleLR', 'CyclicLR')


def build_lr_scheduler(config, optimizer, steps_per_epoch):
    """
    The learning rate scheduler of the `lr_scheduler` block, any class of torch.optim.lr_scheduler.

    `interval` says when it is stepped: after every 'epoch', or after every 'batch'. Schedules with a warmup
    need 'batch', and it is the default for OneCycleLR and CyclicLR. The length of the schedule is filled in
    from the run where it is not given: `epochs` and `steps_per_epoch` of OneCycleLR, and `T_max` of
    CosineAnnealingLR.

    :param steps_per_epoch: number of training batches of an epoch
    :return: (scheduler, interval), or (None, 'epoch') if the config has no scheduler
    """
    cfg = config.get('lr_scheduler')
    if not cfg:
        return None, 'epoch'

    name = cfg['type']
    interval = cfg.get('interval', 'batch' if name in _BATCH_SCHEDULERS else 'epoch')
    if interval not in ('epoch', 'batch'):
        raise ValueError(f"Unknown lr_scheduler interval '{interval}', use 'epoch' or 'batch'.")

    args = dict(cfg.get('args', {}))
    epochs = config['trainer']['epochs']
    if name == 'OneCycleLR':
        if interval != 'batch':
            raise ValueError("OneCycleLR plans every training step, its interval has to be 'batch'.")
        if 'total_steps' not in args:
            args.setdefault('epochs', epochs)
            args.setdefault('steps_per_epoch', steps_per_epoch)
    elif name == 'CosineAnnealingLR':
        args.setdefault('T_max', epochs * steps_per_epoch if interval == 'batch' else epochs)

    scheduler = getattr(torch.optim.lr_scheduler, name)(optimizer, **args)
    return scheduler, interval


def build_style_head(config):
    """
    The style head, configured by the `style_head` block.

    :return: dict of Trainer arguments (style_weight, style_detach), empty if it is not configured
    """
    cfg = config.get('style_head') or {}
    weight = float(cfg.get('weight', 0.0))
    if not weight:
        return {}
    return {'style_weight': weight, 'style_detach': bool(cfg.get('detach', False))}


def build_reconstruction(config):
    """
    Which glyphs the reconstruction is trained on, configured by the `reconstruction` block.

    :return: dict of Trainer arguments (target_glyphs, glyphs_per_sample), empty if it is not configured
    """
    cfg = config.get('reconstruction') or {}
    arguments = {}
    if 'glyphs' in cfg:
        arguments['target_glyphs'] = cfg['glyphs']
    if 'glyphs_per_sample' in cfg:
        arguments['glyphs_per_sample'] = cfg['glyphs_per_sample']
    return arguments


def build_adversarial(config, fonts, device):
    """
    The discriminator of the adversarial loss, configured by the `adversarial_loss` block.

    :return: dict of Trainer arguments (discriminator, discriminator_optimizer, adversarial_weight,
             adversarial_start_epoch), empty if it is not configured
    """
    cfg = config.get('adversarial_loss') or {}
    weight = float(cfg.get('weight', 0.0))
    if not weight:
        return {}

    discriminator = module_arch.GlyphDiscriminator(
        fonts.num_glyphs, base_channels=cfg.get('discriminator_channels', 32)).to(device)
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=cfg.get('lr', 2e-4), betas=(0.5, 0.999))
    return {
        'discriminator': discriminator,
        'discriminator_optimizer': optimizer,
        'adversarial_weight': weight,
        'adversarial_start_epoch': cfg.get('start_epoch', 1),
    }


def build_metrics(config):
    return [getattr(module_metric, met) for met in config['metrics']]


def _pin_memory(requested, device):
    # pinned memory only speeds up transfers to CUDA, other backends warn about it or ignore it
    return bool(requested) and device is not None and torch.device(device).type == 'cuda'


def _as_tuple(value):
    return tuple(value) if isinstance(value, (list, tuple)) else value
