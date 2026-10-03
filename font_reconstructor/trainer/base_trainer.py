from abc import abstractmethod
from pathlib import Path

import torch
from numpy import inf

from font_reconstructor.logger import TensorboardWriter
from font_reconstructor.utils import unwrap_model


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        # the model without its DataParallel wrapper. Checkpoints hold its weights, so they load on any device setup.
        self.core_model = unwrap_model(model)
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')
        # number of periodic checkpoints kept on disk, older ones are deleted. 0 keeps all of them.
        self.keep_last_checkpoints = cfg_trainer.get('keep_last_checkpoints', 5) or 0
        self._saved_checkpoints = []

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1
        self._extras_to_load = {}

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            # set step of tensorboard writer
            self.writer.set_step(epoch)

            result = self._train_epoch(epoch)

            # add optimizer lr to tensorboard
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'])
            # add train and validation metrics in the same plot
            for key, value in result.items():
                self.writer.add_scalar(key.removeprefix('val_') + '/' + ('val' if key.startswith('val_') else 'train'), value)
            # add histogram of model parameters to the tensorboard
            for name, p in self.core_model.named_parameters():
                self.writer.add_histogram(name, p, bins='auto')

            # save logged informations into log dict
            log = {
                'epoch': epoch,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

            periodic = epoch % self.save_period == 0
            if periodic or best:
                self._save_checkpoint(epoch, save_periodic=periodic, save_best=best)

            if self.mnt_mode != 'off' and not_improved_count > self.early_stop:
                self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                 "Training stops.".format(self.early_stop))
                break

    def _save_checkpoint(self, epoch, save_periodic=True, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param save_periodic: if True, save the checkpoint as 'checkpoint-epoch{epoch}.pth'
        :param save_best: if True, save the checkpoint as 'model_best.pth'
        """
        state = {
            'arch': type(self.core_model).__name__,
            'epoch': epoch,
            'state_dict': self.core_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            # the plain config dict, so that loading a checkpoint does not depend on the classes of this project
            'config': _to_plain(self.config.config),
            'extras': self._checkpoint_extras(),
        }
        if save_periodic:
            filename = self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch)
            torch.save(state, str(filename))
            self.logger.info("Saving checkpoint: {} ...".format(filename))
            self._saved_checkpoints.append(filename)
            self._prune_checkpoints()
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _checkpoint_extras(self):
        """
        Further state of a trainer to store in checkpoints, as a dict of plain data and tensors.

        A resumed trainer finds the stored dict in `self._extras_to_load`. It is read there, not passed to a
        hook, because checkpoints are loaded before a subclass has finished setting itself up.
        """
        return {}

    def _prune_checkpoints(self):
        """
        delete the oldest periodic checkpoints written by this run, 'model_best.pth' is never deleted
        """
        if self.keep_last_checkpoints <= 0:
            return

        while len(self._saved_checkpoints) > self.keep_last_checkpoints:
            oldest = Path(self._saved_checkpoints.pop(0))
            if oldest.exists():
                oldest.unlink()

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location='cpu')
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.core_model.load_state_dict(strip_data_parallel_prefix(checkpoint['state_dict']))

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self._extras_to_load = checkpoint.get('extras') or {}

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


def strip_data_parallel_prefix(state_dict):
    """
    remove the 'module.' prefix that checkpoints of DataParallel models carry
    """
    if state_dict and all(key.startswith('module.') for key in state_dict):
        return {key[len('module.'):]: value for key, value in state_dict.items()}
    return state_dict


def _to_plain(value):
    """
    convert nested mappings and sequences to plain dicts and lists
    """
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value
