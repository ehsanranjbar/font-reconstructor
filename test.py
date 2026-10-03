import argparse

import torch

from font_reconstructor import factory
from font_reconstructor.config import ConfigParser
from font_reconstructor.evaluation import build_topk_accuracy, evaluate
from font_reconstructor.trainer import strip_data_parallel_prefix
from font_reconstructor.utils import prepare_device, seed_everything


def main(config):
    """
    Evaluate a checkpoint on clean (not augmented) samples.

    The validation split of the training data is used. If the config holds out no validation split, the
    whole dataset is used instead.

    :return: dict of the loss, metrics and top-k accuracies
    """
    logger = config.get_logger('test')
    assert config.resume is not None, "A checkpoint needs to be specified. Add '-r path/to/checkpoint.pth', for example."

    seed_everything(config.get('seed', 42))
    device, _ = prepare_device(config['n_gpu'], config.get('device', 'auto'))
    logger.info('Using device: {}'.format(device))

    # setup data_loader instances
    fonts = factory.build_fontset(config)
    train_loader, valid_loader = factory.build_train_valid_loaders(
        config, fonts, device, shuffle=False, random_augmentations=False)
    data_loader = valid_loader if valid_loader is not None else train_loader
    clustering_data_loader = factory.build_clustering_loader(config, fonts, device)

    # build model architecture
    model = factory.build_model(config, fonts)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = factory.build_criterion(config)
    metric_fns = factory.build_metrics(config)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, map_location='cpu')
    model.load_state_dict(strip_data_parallel_prefix(checkpoint['state_dict']))

    # prepare model for testing
    model = model.to(device)
    model.eval()

    topk_acc = None
    if clustering_data_loader is not None:
        topk_acc = build_topk_accuracy(model, clustering_data_loader, device, len(fonts))

    log = evaluate(
        model, data_loader, loss_fn, metric_fns, device,
        topk_acc=topk_acc, ks=config['trainer'].get('topk', (5, 10)), desc='Test',
    )
    logger.info(log)
    return log


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Test the font reconstructor')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of CUDA GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
