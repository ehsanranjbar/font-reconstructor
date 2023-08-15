import math

import arabic_reshaper
import matplotlib.pyplot as plt
import numpy as np
import torch
from bidi.algorithm import get_display
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from base import BaseTrainer
from utils import MetricTracker, inf_loop, move_model_to_cpu


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        train_loop = tqdm(self.data_loader, desc=f'Epoch [{epoch}]')
        for batch_idx, (data, target, _, _) in enumerate(train_loop):
            # write the model graph at first batch of epoch 1
            if epoch == 1 and batch_idx == 0:
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

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(log['val_loss'])
            else:
                self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        embedings = []
        fonts = []
        with torch.no_grad():
            for batch_idx, (data, target, text, font) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                latent = self.model.encoder(data)
                output = self.model.decoder(latent)
                loss = self.criterion(output, target)

                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

                if epoch % math.ceil(self.epochs / 8) == 0:
                    embedings.append(latent)
                    fonts.extend(font)

                if batch_idx == 0:
                    self._samples_figure(data, target, latent, output, text, font)

            # add embedding to tensorboard
            # we can only add embedding 8 times to the tensorboard so we do this every (self.epochs / 8)
            if epoch % math.ceil(self.epochs / 8) == 0:
                embedings = torch.cat(embedings, dim=0)
                self.writer.add_embedding(embedings, metadata=fonts, global_step=epoch)

        return self.valid_metrics.result()

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
        self.writer.add_figure('samples', self._plot_samples(
            data.cpu().numpy(),
            target.cpu().numpy(),
            latent.cpu().numpy(),
            output.cpu().numpy(),
            text,
            font,
            loss,
        ))

    def _plot_samples(self, data, target, latent, output, text, font, loss):
        # create figure
        fig = plt.figure(figsize=(20, 20))
        subfigs = fig.subfigures(nrows=data.shape[0], ncols=1)
        for i, subfig in enumerate(subfigs):
            subfig.suptitle('Text: \"{}\", Font: \"{}\", Loss: {:.4f}'.
                            format(get_display(arabic_reshaper.reshape(text[i])),
                                   font[i],
                                   loss[i]))
            axs = subfig.subplots(nrows=1, ncols=4, gridspec_kw={'width_ratios': [1, 5, 1, 1]})

            # plot original image
            axs[0].imshow(data[i, 0], cmap='gray')
            axs[0].set_title('Input')

            # plot output under the target font fingerprints after concatenating characters in each channel side by side
            tgt = np.concatenate(target[i], axis=1)
            out = np.concatenate(output[i], axis=1)
            tgt_out = np.concatenate([tgt, out], axis=0)
            axs[1].imshow(tgt_out, cmap='gray')
            axs[1].set_title('Target vs Output')

            # plot the latent space of encoder in square sized 2d array
            log_size = int(math.log2(latent[i].shape[0]))
            squared_latent = np.reshape(
                latent[i], (2**int(log_size / 2), 2**(log_size - int(log_size / 2))))
            im = axs[2].imshow(squared_latent)
            divider = make_axes_locatable(axs[2])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
            axs[2].set_title('Latent Space')

            # plot histogram of latent space
            _, bins, patches = axs[3].hist(latent[i], bins='auto')
            fracs = bins / bins.max()
            norm = colors.Normalize(fracs.min(), fracs.max())
            for f, p in zip(fracs, patches):
                c = plt.cm.viridis(norm(f))
                p.set_facecolor(c)

        return fig
