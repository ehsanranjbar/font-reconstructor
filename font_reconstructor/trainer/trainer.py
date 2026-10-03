import math

import torch
import torch.nn.functional as F
from tqdm import tqdm

from font_reconstructor.evaluation import build_topk_accuracy, evaluate
from font_reconstructor.model.loss import adversarial_loss, discriminator_loss
from font_reconstructor.model.metric import style_accuracy
from font_reconstructor.reporting import ValidationReporter, base_dataset
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
    :param contrastive_criterion: optional loss function of (latent, font_index) that shapes the latent space
        for font retrieval. It is added to the reconstruction loss, scaled by `contrastive_weight`.

    :param discriminator: optional GlyphDiscriminator. With it the decoder is also trained to draw fingerprints
        that the discriminator can not tell from real ones, scaled by `adversarial_weight`. The discriminator
        is trained alongside with `discriminator_optimizer`.
    :param adversarial_start_epoch: first epoch with the adversarial term. Before it the decoder learns the
        rough shapes from the reconstruction loss alone, which gives the discriminator something to refine.
    :param style_weight: weight of the cross-entropy loss of the model's style head. With it, `style_loss` and
        `style_acc` are logged for training, and `style_acc` and `style_balanced_acc` for validation.
    :param style_detach: train the style head on the latent vector without letting it shape the encoder

    The logged `loss` is the reconstruction loss in training and in validation, so the two are comparable.
    The other terms and their weighted sum `total_loss` are logged for training too.

    Validation also logs the scalars of ValidationReport, and with tensorboard enabled it writes the figures of
    ValidationReporter every `figure_period` epochs (a setting of the trainer config, 1 by default).
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, clustering_data_loader=None, num_fonts=None,
                 lr_scheduler=None, len_epoch=None, contrastive_criterion=None, contrastive_weight=1.0,
                 discriminator=None, discriminator_optimizer=None, adversarial_weight=0.0,
                 adversarial_start_epoch=1, style_weight=0.0, style_detach=False):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.style_weight = style_weight
        self.style_detach = style_detach
        if self.style_weight and getattr(self.core_model, 'style_head', None) is None:
            raise ValueError("style_weight is set, but the model has no style head.")
        self.contrastive_criterion = contrastive_criterion if contrastive_weight else None
        self.contrastive_weight = contrastive_weight

        self.discriminator = discriminator if adversarial_weight else None
        self.discriminator_optimizer = discriminator_optimizer
        self.adversarial_weight = adversarial_weight
        self.adversarial_start_epoch = adversarial_start_epoch
        if self.discriminator is not None:
            if self.discriminator_optimizer is None:
                raise ValueError("A discriminator needs a discriminator_optimizer.")
            # state of a resumed run, see BaseTrainer._checkpoint_extras
            if 'discriminator' in self._extras_to_load:
                self.discriminator.load_state_dict(self._extras_to_load['discriminator'])
                self.discriminator_optimizer.load_state_dict(self._extras_to_load['discriminator_optimizer'])
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

        # the fixed panel of samples and the baseline that every validation is compared on
        self.figure_period = max(1, int(config['trainer'].get('figure_period', 1)))
        self.reporter = None
        if self.do_validation:
            self.reporter = ValidationReporter(self.valid_data_loader.dataset, data_loader.dataset)

        extra_keys = []
        if self.contrastive_criterion is not None:
            extra_keys.append('contrastive_loss')
        if self.discriminator is not None:
            extra_keys.extend(['adversarial_loss', 'discriminator_loss'])
        if self.style_weight:
            extra_keys.extend(['style_loss', 'style_acc'])
        if extra_keys:
            extra_keys.append('total_loss')
        self.train_metrics = MetricTracker('loss', *extra_keys, *[m.__name__ for m in self.metric_ftns])

    def _checkpoint_extras(self):
        if self.discriminator is None:
            return {}
        return {
            'discriminator': self.discriminator.state_dict(),
            'discriminator_optimizer': self.discriminator_optimizer.state_dict(),
        }

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        adversarial = self.discriminator is not None and epoch >= self.adversarial_start_epoch
        if self.discriminator is not None:
            self.discriminator.train()
        batches = self.data_loader if self._batches is None else self._batches
        train_loop = tqdm(batches, total=self.len_epoch, desc=f'Epoch [{epoch}]')
        for batch_idx, batch in enumerate(train_loop):
            data, target = batch['image'], batch['target']

            # write the model graph at first batch of epoch 1
            if epoch == 1 and batch_idx == 0 and self.writer.enabled:
                self.writer.add_graph(move_model_to_cpu(self.model), input_to_model=data, verbose=False)

            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output, latent = self.model(data, return_latent=True)
            loss = self.criterion(output, target)
            total_loss = loss
            if self.contrastive_criterion is not None:
                contrastive_loss = self.contrastive_criterion(latent, batch['font_index'].to(self.device))
                total_loss = total_loss + self.contrastive_weight * contrastive_loss
                self.train_metrics.update('contrastive_loss', contrastive_loss.item())
            if self.style_weight:
                style_index = batch['style_index'].to(self.device)
                logits = self.core_model.predict_style(latent.detach() if self.style_detach else latent)
                style_loss = F.cross_entropy(logits, style_index)
                total_loss = total_loss + self.style_weight * style_loss
                self.train_metrics.update('style_loss', style_loss.item())
                self.train_metrics.update('style_acc', style_accuracy(logits, style_index))
            if adversarial:
                self.train_metrics.update('discriminator_loss', self._train_discriminator(output.detach(), target))
                generator_loss = adversarial_loss(self.discriminator(output))
                total_loss = total_loss + self.adversarial_weight * generator_loss
                self.train_metrics.update('adversarial_loss', generator_loss.item())
            if total_loss is not loss:
                self.train_metrics.update('total_loss', total_loss.item())
            total_loss.backward()
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

    def _train_discriminator(self, output, target):
        """
        one step of the discriminator on real fingerprints and on those the decoder just drew

        :return: the discriminator loss as a float
        """
        self.discriminator_optimizer.zero_grad()
        loss = discriminator_loss(self.discriminator(target), self.discriminator(output))
        loss.backward()
        self.discriminator_optimizer.step()
        return loss.item()

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
        embeddings, embedded_fonts = [], []

        def on_batch(batch_idx, batch, data, target, latent, output):
            if log_embeddings:
                embeddings.append(latent.cpu())
                embedded_fonts.extend(batch['font_index'].tolist())

        report = self.reporter.new_report()
        val_log = evaluate(
            self.core_model, self.valid_data_loader, self.criterion, self.metric_ftns, self.device,
            topk_acc=topk_acc, ks=self.topk, on_batch=on_batch, report=report,
        )

        if self.writer.enabled and epoch % self.figure_period == 0:
            for tag, figure in self.reporter.figures(self.core_model, self.device, report, topk_acc).items():
                self.writer.add_figure(tag, figure)

        # add embedding to tensorboard, with the font, its family and its style to color the points by
        if log_embeddings and embeddings:
            fonts = getattr(base_dataset(self.valid_data_loader.dataset), 'fonts', None)
            if fonts is not None:
                metadata = [[fonts.names[i], fonts.families[i], fonts.styles[i]] for i in embedded_fonts]
                self.writer.add_embedding(torch.cat(embeddings, dim=0), metadata=metadata,
                                          metadata_header=['font', 'family', 'style'], global_step=epoch)
            else:
                self.writer.add_embedding(torch.cat(embeddings, dim=0), metadata=embedded_fonts, global_step=epoch)

        return val_log
