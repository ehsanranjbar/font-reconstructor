import argparse
import collections

import torch

from font_reconstructor import factory
from font_reconstructor.config import ConfigParser
from font_reconstructor.trainer import Trainer
from font_reconstructor.utils import prepare_device, seed_everything


def main(config):
    logger = config.get_logger('train')

    # fix random seeds for reproducibility
    seed_everything(config.get('seed', 42))

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'], config.get('device', 'auto'))
    logger.info('Using device: {}'.format(device))

    # setup data_loader instances
    fonts = factory.build_fontset(config)
    corpus = factory.build_corpus(config)
    data_loader, valid_data_loader = factory.build_train_valid_loaders(config, fonts, device, corpus)
    clustering_data_loader = factory.build_clustering_loader(config, fonts, device, corpus)
    n_valid_fonts = 0 if valid_data_loader is None else len(valid_data_loader.dataset.dataset.font_indices)
    logger.info('Fonts: {} for training, {} held out for validation'.format(
        len(data_loader.dataset.dataset.font_indices), n_valid_fonts))
    logger.info('Styles: {}'.format(', '.join(
        '{} ({})'.format(name, fonts.styles.count(name)) for name in fonts.style_names)))

    # build model architecture, then print to console
    model = factory.build_model(config, fonts)
    factory.check_model_shapes(model, data_loader.dataset)
    logger.info(model)

    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = factory.build_criterion(config)
    contrastive_criterion, contrastive_weight = factory.build_contrastive(config)
    metrics = factory.build_metrics(config)

    # build optimizer and learning rate scheduler. Remove the lr_scheduler block of the config to train without one.
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler, lr_scheduler_interval = factory.build_lr_scheduler(config, optimizer, len(data_loader))

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      clustering_data_loader=clustering_data_loader,
                      num_fonts=len(fonts),
                      lr_scheduler=lr_scheduler,
                      lr_scheduler_interval=lr_scheduler_interval,
                      contrastive_criterion=contrastive_criterion,
                      contrastive_weight=contrastive_weight,
                      **factory.build_adversarial(config, fonts, device),
                      **factory.build_style_head(config))

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Train the font reconstructor')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of CUDA GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
