"""
Learning rate range test: find the learning rates a config can train with, in a few minutes.

The model of the config is trained for a few hundred steps while the learning rate rises exponentially from
--min-lr to --max-lr (Smith, "Cyclical Learning Rates for Training Neural Networks", 2017). At low rates the
loss barely moves, then it falls, and past some rate the steps get too large and it rises again. The test
stops once the loss has clearly turned upward.

It prints two readings of the smoothed loss and writes a plot and a csv file:

  steepest   the rate where the loss falls fastest. A safe constant learning rate.
  minimum    the rate where the loss is lowest. Training is on the edge of diverging there, so the usual
             choice for the peak of a one-cycle schedule is a few times below it.

Read the plot as well. If the loss is flat over a wide range before it rises, the minimum can lie anywhere on
that plateau, and the rate to pick is the one where the loss stops falling.

Nothing is saved from the model. Use the same config as for training, with its batch size: the usable rates
change with the batch size.

Usage: python scripts/find_lr.py -c config.json [--steps 400] [--min-lr 1e-5] [--max-lr 1]
"""
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from font_reconstructor import factory  # noqa: E402
from font_reconstructor.config import ConfigParser  # noqa: E402
from font_reconstructor.logger import figures  # noqa: E402
from font_reconstructor.trainer import Trainer  # noqa: E402
from font_reconstructor.utils import prepare_device, read_json, seed_everything  # noqa: E402


def smooth(values, beta=0.9):
    """
    exponential moving average with bias correction, so that the first values are not pulled towards zero
    """
    smoothed, average = [], 0.0
    for step, value in enumerate(values, start=1):
        average = beta * average + (1 - beta) * value
        smoothed.append(average / (1 - beta ** step))
    return np.array(smoothed)


def read_marks(rates, smoothed, skip_start=10, skip_end=5):
    """
    :return: dict with the learning rate of the steepest fall and of the minimum of the smoothed loss
    """
    rates, smoothed = np.asarray(rates), np.asarray(smoothed)
    end = max(len(smoothed) - skip_end, skip_start + 2)
    slope = np.gradient(smoothed, np.log10(rates[:len(smoothed)]))
    steepest = skip_start + int(np.argmin(slope[skip_start:end]))
    minimum = skip_start + int(np.argmin(smoothed[skip_start:end]))
    return {'steepest': float(rates[steepest]), 'minimum': float(rates[minimum])}


def range_test(config, steps, min_lr, max_lr, stop_factor=4.0):
    """
    :return: (rates, losses) of the steps that were run. losses is a dict of name to list.
    """
    seed_everything(config.get('seed', 42))
    device, _ = prepare_device(config['n_gpu'], config.get('device', 'auto'))
    print(f"Using device: {device}")

    fonts = factory.build_fontset(config)
    corpus = factory.build_corpus(config)
    data_loader, _ = factory.build_train_valid_loaders(config, fonts, device, corpus)
    model = factory.build_model(config, fonts).to(device)
    contrastive_criterion, contrastive_weight = factory.build_contrastive(config)
    optimizer = config.init_obj('optimizer', torch.optim, model.parameters())

    # the trainer is only used for its loss, with everything of the config that goes into it
    trainer = Trainer(model, factory.build_criterion(config), [], optimizer, config=config, device=device,
                      data_loader=data_loader, contrastive_criterion=contrastive_criterion,
                      contrastive_weight=contrastive_weight, **factory.build_style_head(config))

    rates = min_lr * (max_lr / min_lr) ** (np.arange(steps) / max(steps - 1, 1))
    losses, best, batches = {}, float('inf'), iter(data_loader)
    model.train()
    for step, rate in enumerate(rates):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(data_loader)
            batch = next(batches)
        for group in optimizer.param_groups:
            group['lr'] = float(rate)

        optimizer.zero_grad()
        step_losses, _, _ = trainer.compute_losses(batch)
        step_losses['total_loss'].backward()
        optimizer.step()

        for name, value in step_losses.items():
            if name.endswith('loss'):
                losses.setdefault(name, []).append(float(value.detach()) if torch.is_tensor(value) else float(value))
        current = smooth(losses['total_loss'])[-1]
        best = min(best, current)
        if step % 25 == 0:
            print(f"step {step:4d}  lr {rate:.2e}  total loss {losses['total_loss'][-1]:.4f}")
        if not np.isfinite(current) or (step > 20 and current > stop_factor * best):
            print(f"The loss turned upward at lr {rate:.2e}, stopping after {step + 1} steps.")
            break

    return rates[:len(losses['total_loss'])], losses


def main():
    parser = argparse.ArgumentParser(description='Learning rate range test')
    parser.add_argument('-c', '--config', required=True, help='config file path')
    parser.add_argument('--steps', default=400, type=int, help='number of training steps (default: 400)')
    parser.add_argument('--min-lr', default=1e-5, type=float, help='learning rate of the first step (default: 1e-5)')
    parser.add_argument('--max-lr', default=1.0, type=float, help='learning rate of the last step (default: 1)')
    args = parser.parse_args()

    config = read_json(args.config)
    config['trainer']['tensorboard'] = False
    config.pop('adversarial_loss', None)
    run_id = 'lr_range_test_' + datetime.now().strftime(r'%m%d_%H%M%S')
    config = ConfigParser(config, run_id=run_id)

    rates, losses = range_test(config, args.steps, args.min_lr, args.max_lr)
    smoothed = smooth(losses['total_loss'])
    marks = read_marks(rates, smoothed)

    out_dir = Path(config.log_dir)
    with open(out_dir / 'lr_range_test.csv', 'wt', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['learning_rate', 'smoothed_total_loss', *losses])
        for row in zip(rates, smoothed, *losses.values()):
            writer.writerow(row)
    figure = figures.plot_lr_range_test(rates, losses, smoothed, marks)
    FigureCanvasAgg(figure)
    figure.savefig(out_dir / 'lr_range_test.png')

    print(f"steepest fall of the loss at lr {marks['steepest']:.2e}")
    print(f"lowest loss at lr {marks['minimum']:.2e}")
    print(f"a peak learning rate of about {marks['minimum'] / 4:.1e} suits a one-cycle schedule")
    print(f"plot and values: {out_dir}")


if __name__ == '__main__':
    main()
