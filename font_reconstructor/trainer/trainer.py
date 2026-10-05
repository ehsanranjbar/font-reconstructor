import collections
import math
import shutil

import torch
import torch.nn.functional as F
from tqdm import tqdm

from font_reconstructor.evaluation import build_topk_accuracy, evaluate
from font_reconstructor.logger.figures import save_figure
from font_reconstructor.model.loss import adversarial_loss, discriminator_loss
from font_reconstructor.model.model import select_glyphs
from font_reconstructor.model.metric import style_accuracy
from font_reconstructor.reporting import ValidationReporter, base_dataset
from font_reconstructor.utils import MetricTracker, inf_loop, move_model_to_cpu

from .base_trainer import BaseTrainer


def pick_glyphs(seen, count):
    """
    Pick `count` glyphs of each sample at random among those its text shows.

    A sample that shows fewer than `count` glyphs gets them all and some of them again. One that shows none
    gets a single arbitrary glyph, so that the result always has the same shape.

    :param seen: bool tensor (batch, glyphs), True for the glyphs that occur in the text of a sample
    :return: tensor (batch, count) of glyph indices
    """
    # a random order of the glyphs of each sample, with those of its text first
    order = (torch.rand(seen.shape, device=seen.device) + seen).argsort(dim=1, descending=True)
    available = seen.sum(dim=1, keepdim=True).clamp(min=1)
    positions = torch.arange(count, device=seen.device).unsqueeze(0) % available
    return order.gather(1, positions)


class Trainer(BaseTrainer):
    """
    Trainer class

    All data loaders yield sample dicts, see RandomTextImageDataset.

    :param clustering_data_loader: loader used to estimate the centre of each font in the latent space before
        validation. Without it validation reports no top-k accuracy.
    :param num_fonts: number of fonts, needed together with clustering_data_loader
    :param len_epoch: number of batches of an epoch for iteration-based training. One pass over data_loader if None.
    :param lr_scheduler_interval: 'epoch' to step the learning rate scheduler after every epoch, 'batch' to step
        it after every batch, which schedules with a warmup need. The scheduler is stored in checkpoints.
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
    :param target_glyphs: 'all' to train the reconstruction on the whole fingerprint, 'text' to train it only
        on glyphs that occur in the text of each sample. The model is not asked to guess glyphs it has not
        seen then, and a conditioned decoder only draws a few glyphs of each sample, which makes it affordable.
        Validation always draws the whole fingerprint.
    :param glyphs_per_sample: number of glyphs drawn for each sample with target_glyphs 'text'. They are picked
        at random from the glyphs of its text. A text with fewer glyphs gets some of them twice, so that every
        batch has the same shape and every sample the same weight.

    The logged `loss` is the reconstruction loss in training and in validation, so the two are comparable.
    With `target_glyphs` 'text' the training loss covers the glyphs of the text only: compare it with
    `val_seen_glyph_loss`, which is a plain pixel error and not the configured loss, so only roughly.
    The other terms and their weighted sum `total_loss` are logged for training too.

    Validation also logs the scalars of ValidationReport, and every `figure_period` epochs it draws the figures of
    ValidationReporter. They go to tensorboard if that is enabled, and with `save_figures` they are written as
    image files to `figures/` in the log directory of the run, at `figure_dpi` dots per inch. These are settings
    of the trainer config.
    """

    _RECENT_STEPS = 100  # steps that the loss on the progress bar is averaged over
    # the step of the run, the time that is left and the loss, nothing else
    _BAR_FORMAT = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{remaining} left{postfix}]'

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, clustering_data_loader=None, num_fonts=None,
                 lr_scheduler=None, lr_scheduler_interval='epoch', len_epoch=None,
                 contrastive_criterion=None, contrastive_weight=1.0,
                 discriminator=None, discriminator_optimizer=None, adversarial_weight=0.0,
                 adversarial_start_epoch=1, style_weight=0.0, style_detach=False, target_glyphs='all',
                 glyphs_per_sample=8):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        if target_glyphs not in ('all', 'text'):
            raise ValueError(f"Unknown target_glyphs '{target_glyphs}', use 'all' or 'text'.")
        if target_glyphs == 'text' and discriminator is not None and adversarial_weight:
            raise ValueError("The adversarial loss judges whole fingerprints, it needs target_glyphs 'all'.")
        self.target_glyphs = target_glyphs
        self.glyphs_per_sample = int(glyphs_per_sample)
        if self.glyphs_per_sample < 1:
            raise ValueError("glyphs_per_sample has to be at least 1.")
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
        if lr_scheduler_interval not in ('epoch', 'batch'):
            raise ValueError(f"Unknown lr_scheduler_interval '{lr_scheduler_interval}', use 'epoch' or 'batch'.")
        self.lr_scheduler_interval = lr_scheduler_interval
        if self.lr_scheduler is not None and 'lr_scheduler' in self._extras_to_load:
            # a resumed run continues its schedule instead of starting it again
            self.lr_scheduler.load_state_dict(self._extras_to_load['lr_scheduler'])

        # the fixed panel of samples and the baseline that every validation is compared on
        self.figure_period = max(1, int(config['trainer'].get('figure_period', 1)))
        self.save_figures = bool(config['trainer'].get('save_figures', True))
        self.figure_dpi = int(config['trainer'].get('figure_dpi', 200))
        self.figure_dir = config.log_dir / 'figures'
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
        extras = {}
        if self.lr_scheduler is not None:
            extras['lr_scheduler'] = self.lr_scheduler.state_dict()
        if self.discriminator is not None:
            extras['discriminator'] = self.discriminator.state_dict()
            extras['discriminator_optimizer'] = self.discriminator_optimizer.state_dict()
        return extras

    def compute_losses(self, batch, adversarial=False):
        """
        The losses of one training batch. With `adversarial` the discriminator is trained one step on the way.

        :param batch: a sample dict of the training loader
        :return: (losses, output, target). losses is a dict of tensors: `loss` is the reconstruction loss,
                 `total_loss` the weighted sum that is optimized, and the other terms are there if they are
                 configured. `style_acc` is a float.
        """
        data, target = batch['image'].to(self.device), batch['target'].to(self.device)
        if self.target_glyphs == 'text':
            # only some glyphs of each text are drawn and compared
            glyphs = pick_glyphs(batch['seen'].to(self.device), self.glyphs_per_sample)
            output, latent = self.model(data, return_latent=True, glyphs=glyphs)
            target = select_glyphs(target, glyphs)
            losses = {'loss': self.criterion(output, target)}
        else:
            output, latent = self.model(data, return_latent=True)
            losses = {'loss': self.criterion(output, target)}
        total_loss = losses['loss']
        if self.contrastive_criterion is not None:
            losses['contrastive_loss'] = self.contrastive_criterion(latent, batch['font_index'].to(self.device))
            total_loss = total_loss + self.contrastive_weight * losses['contrastive_loss']
        if self.style_weight:
            style_index = batch['style_index'].to(self.device)
            logits = self.core_model.predict_style(latent.detach() if self.style_detach else latent)
            losses['style_loss'] = F.cross_entropy(logits, style_index)
            losses['style_acc'] = style_accuracy(logits, style_index)
            total_loss = total_loss + self.style_weight * losses['style_loss']
        if adversarial:
            losses['discriminator_loss'] = self._train_discriminator(output.detach(), target)
            losses['adversarial_loss'] = adversarial_loss(self.discriminator(output))
            total_loss = total_loss + self.adversarial_weight * losses['adversarial_loss']
        losses['total_loss'] = total_loss
        return losses, output, target

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        # a sampler whose epochs draw from different parts of the data has to know which epoch this is
        batch_sampler = getattr(self.data_loader, 'batch_sampler', None)
        if hasattr(batch_sampler, 'set_epoch'):
            batch_sampler.set_epoch(epoch)
        adversarial = self.discriminator is not None and epoch >= self.adversarial_start_epoch
        if self.discriminator is not None:
            self.discriminator.train()
        batches = self.data_loader if self._batches is None else self._batches
        # The bar counts the steps of the whole run, so that it shows how far the run is and when it ends
        # also where an epoch is only a part of one long pass over the data.
        steps_before = (epoch - 1) * self.len_epoch
        # it is advanced by hand after every step: wrapped around the batches it would miss the last step of
        # an epoch, which leaves the loop before the bar counts it
        train_loop = tqdm(total=self.epochs * self.len_epoch, initial=steps_before,
                          desc=f'Epoch {epoch}/{self.epochs}', bar_format=self._BAR_FORMAT)
        shown_loss = 'total_loss' if 'total_loss' in self.train_metrics.keys else 'loss'
        recent_losses = collections.deque(maxlen=self._RECENT_STEPS)
        for batch_idx, batch in enumerate(batches):
            # write the model graph at first batch of epoch 1
            if epoch == 1 and batch_idx == 0 and self.writer.enabled:
                self.writer.add_graph(move_model_to_cpu(self.model), input_to_model=batch['image'], verbose=False)

            self.optimizer.zero_grad()
            losses, output, target = self.compute_losses(batch, adversarial)
            losses['total_loss'].backward()
            self.optimizer.step()
            if self.lr_scheduler is not None and self.lr_scheduler_interval == 'batch':
                self.lr_scheduler.step()

            for name in self.train_metrics.keys:
                if name in losses:
                    value = losses[name]
                    self.train_metrics.update(name, value.item() if torch.is_tensor(value) else value)
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            # The progress bar shows the loss that is optimized, averaged over the last steps: an average over
            # a long epoch would mostly tell how its first steps went.
            recent_losses.append(losses[shown_loss].item())
            train_loop.set_postfix(loss='{:.4f}'.format(sum(recent_losses) / len(recent_losses)), refresh=False)
            train_loop.update(1)

            if batch_idx + 1 >= self.len_epoch:
                break
        train_loop.close()

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None and self.lr_scheduler_interval == 'epoch':
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # without a validation set the plateau is judged on the training loss
                self.lr_scheduler.step(log.get('val_loss', log['loss']))
            else:
                self.lr_scheduler.step()
        return log

    def _save_figure(self, tag, figure, epoch):
        """
        Write a figure into the log directory of the run, twice: under its own name with the epoch, so that the
        epochs of one figure sit side by side, and under `latest`, which always holds the newest of each figure.
        """
        name = tag.replace('/', '_')
        path = save_figure(figure, self.figure_dir / name / f'epoch_{epoch:03d}.png', dpi=self.figure_dpi)
        latest = self.figure_dir / 'latest' / f'{name}.png'
        latest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, latest)

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

        if (self.writer.enabled or self.save_figures) and epoch % self.figure_period == 0:
            for tag, figure in self.reporter.figures(self.core_model, self.device, report, topk_acc).items():
                if self.save_figures:
                    self._save_figure(tag, figure, epoch)
                if self.writer.enabled:
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
