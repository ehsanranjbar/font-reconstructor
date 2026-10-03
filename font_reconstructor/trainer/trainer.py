import math

import torch
from tqdm import tqdm

from font_reconstructor.evaluation import build_topk_accuracy, evaluate
from font_reconstructor.logger.figures import plot_samples
from font_reconstructor.utils import MetricTracker, inf_loop, move_model_to_cpu

from .base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class

    All data loaders yield sample dicts, see RandomTextImageDataset.

    :param clustering_data_loader: loader used to estimate the centre of each font in the latent space before
        validation. Without it validation reports no top-k accuracy.
    :param num_fonts: number of fonts, needed together with clustering_data_loader
    :param len_epoch: number of batches of an epoch for iteration-based training. One pass over data_loader if None.
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, clustering_data_loader=None, num_fonts=None,
                 lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
            self._batches = None
        else:
            # iteration-based training
            self.len_epoch = len_epoch
            self._batches = inf_loop(data_loader)
        self.valid_data_loader = valid_data_loader
        self.clustering_data_loader = clustering_data_loader
        self.num_fonts = num_fonts
        self.do_validation = self.valid_data_loader is not None
        self.do_topk = self.do_validation and self.clustering_data_loader is not None
        if self.do_topk and self.num_fonts is None:
            raise ValueError("num_fonts is required when a clustering data loader is given.")
        self.topk = tuple(config['trainer'].get('topk', (5, 10))) if self.do_topk else ()
        self.lr_scheduler = lr_scheduler

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        batches = self.data_loader if self._batches is None else self._batches
        train_loop = tqdm(batches, total=self.len_epoch, desc=f'Epoch [{epoch}]')
        for batch_idx, batch in enumerate(train_loop):
            data, target = batch['image'], batch['target']

            # write the model graph at first batch of epoch 1
            if epoch == 1 and batch_idx == 0 and self.writer.enabled:
                self.writer.add_graph(move_model_to_cpu(self.model), input_to_model=data, verbose=False)

            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            # add stuff to progress bar in the end
            train_loop.set_postfix(loss='{:.4f}'.format(self.train_metrics.avg('loss')))

            if batch_idx + 1 >= self.len_epoch:
                break
        train_loop.close()

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # without a validation set the plateau is judged on the training loss
                self.lr_scheduler.step(log.get('val_loss', log['loss']))
            else:
                self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        topk_acc = None
        if self.do_topk:
            topk_acc = build_topk_accuracy(
                self.core_model, self.clustering_data_loader, self.device, self.num_fonts)

        # we can only add embedding 8 times to the tensorboard so we do this every (self.epochs / 8)
        log_embeddings = self.writer.enabled and epoch % math.ceil(self.epochs / 8) == 0
        epoch_embedings = []
        epoch_fonts = []
        last_batch_idx = len(self.valid_data_loader) - 1

        def on_batch(batch_idx, batch, data, target, latent, output):
            if log_embeddings:
                epoch_embedings.append(latent.cpu())
                epoch_fonts.extend(batch['font'])

            # add figure of samples from validation set at end of each epoch
            if self.writer.enabled and batch_idx == last_batch_idx:
                self._samples_figure(data, target, latent, output, batch['text'], batch['font'])

        val_log = evaluate(
            self.core_model, self.valid_data_loader, self.criterion, self.metric_ftns, self.device,
            topk_acc=topk_acc, ks=self.topk, on_batch=on_batch,
        )

        # add embedding to tensorboard
        if log_embeddings and epoch_embedings:
            self.writer.add_embedding(torch.cat(epoch_embedings, dim=0), metadata=epoch_fonts, global_step=epoch)

        return val_log

    def _samples_figure(self, data, target, latent, output, text, font, n=10):
        # sample the first n images from batch
        n = min(n, data.shape[0])
        data = data[:n]
        target = target[:n]
        latent = latent[:n]
        output = output[:n]
        text = text[:n]
        font = font[:n]

        # get loss of each output
        loss = [self.criterion(o, t).item() for o, t in zip(output, target)]

        # plot and add to tensorboard
        self.writer.add_figure('samples', plot_samples(
            data.cpu().numpy(),
            target.cpu().numpy(),
            latent.cpu().numpy(),
            output.cpu().numpy(),
            text,
            font,
            loss,
        ))
