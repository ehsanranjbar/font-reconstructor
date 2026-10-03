import math

import arabic_reshaper
import matplotlib.pyplot as plt
import numpy as np
from bidi.algorithm import get_display
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_samples(data, target, latent, output, text, font, loss):
    """
    Plot one row per sample: input image, target vs reconstructed fingerprint, latent vector and its histogram.

    :param data: numpy array of input images, shaped (N, 1, H, W)
    :param target: numpy array of target fingerprints, shaped (N, glyphs, H, W)
    :param latent: numpy array of latent vectors, shaped (N, latent_dim)
    :param output: numpy array of reconstructed fingerprints, shaped like target
    :param text: sequence of N rendered texts
    :param font: sequence of N font names
    :param loss: sequence of N per-sample losses
    :return: matplotlib figure
    """
    n = data.shape[0]
    fig = plt.figure(figsize=(20, 2 * n))
    subfigs = np.atleast_1d(fig.subfigures(nrows=n, ncols=1))
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

        # plot the latent vector as a 2d array that is as close to a square as its length allows
        im = axs[2].imshow(_as_grid(latent[i]))
        divider = make_axes_locatable(axs[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        axs[2].set_title('Latent Space')

        # plot histogram of latent space
        _, bins, patches = axs[3].hist(latent[i], bins='auto')
        norm = colors.Normalize(bins.min(), bins.max())
        for b, p in zip(bins, patches):
            p.set_facecolor(plt.cm.viridis(norm(b)))

    return fig


def _as_grid(vector):
    """
    reshape a 1d vector to the most square 2d grid its length divides into
    """
    length = vector.shape[0]
    rows = next(r for r in range(math.isqrt(length), 0, -1) if length % r == 0)
    return np.reshape(vector, (rows, length // rows))
