from collections import OrderedDict
from typing import Tuple

import torch.nn as nn

_ENCODER_LAYERS = 6
_DECODER_LAYERS = 4


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class AutoEncoder(BaseModel):
    """
    Convolutional autoencoder from a text image to the fingerprint of its font.

    The encoder halves the image six times and projects it to a latent vector of `latent_dim`. The decoder
    doubles a small feature map four times up to the fingerprint, one output channel per glyph.

    :param kernel_size: kernel size of all convolutions. The decoder only reaches `output_dims` with 3.
    :param base_conv_filters: filters of the first convolution, doubled by each following encoder layer
    :param batch_norm: add batch normalization after each hidden convolution
    :param latent_dim: size of the latent vector
    :param decoder_output_channels: number of glyphs in the font fingerprint
    :param input_dims: (height, width) of the text image
    :param output_dims: (height, width) of each glyph of the fingerprint
    """

    def __init__(
            self,
            kernel_size: int = 3,
            base_conv_filters: int = 16,
            batch_norm: bool = False,
            latent_dim: int = 128,
            decoder_output_channels: int = 42,
            input_dims: Tuple[int, int] = (32, 128),
            output_dims: Tuple[int, int] = (32, 32),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dims = tuple(input_dims)
        self.output_dims = tuple(output_dims)
        self.output_channels = decoder_output_channels

        def filters(i):
            return base_conv_filters * 2 ** i

        # encoder
        height, width = self.input_dims
        self.encoder = nn.Sequential()
        for i in range(_ENCODER_LAYERS):
            layer = OrderedDict()
            layer['conv2d'] = nn.Conv2d(
                in_channels=1 if i == 0 else filters(i - 1),
                out_channels=filters(i),
                kernel_size=kernel_size,
                stride=2,
                padding=1,
                bias=not batch_norm,
            )
            if batch_norm:
                layer['batch_norm'] = nn.BatchNorm2d(filters(i))
            layer['leaky_relu'] = nn.LeakyReLU(0.2, inplace=True)

            self.encoder.add_module(f'conv_layer_{i}', nn.Sequential(layer))
            height, width = _conv_out(height, kernel_size), _conv_out(width, kernel_size)

        if height < 1 or width < 1:
            raise ValueError(f"input_dims {self.input_dims} are too small for kernel size {kernel_size}.")
        encoder_channels = filters(_ENCODER_LAYERS - 1)
        self.encoder.add_module('flatten', nn.Flatten())
        self.encoder.add_module('fc', nn.Linear(encoder_channels * height * width, latent_dim))

        # decoder, starts from a feature map that the transposed convolutions grow to output_dims
        scale = 2 ** _DECODER_LAYERS
        seed_dims = (self.output_dims[0] // scale, self.output_dims[1] // scale)
        grown_dims = tuple(_grow(n, kernel_size, _DECODER_LAYERS) for n in seed_dims)
        if min(seed_dims) < 1 or grown_dims != self.output_dims:
            raise ValueError(
                f"The decoder can not produce output_dims {self.output_dims} with kernel size {kernel_size}. "
                f"Use kernel size 3 and dimensions that are multiples of {scale}."
            )

        self.decoder = nn.Sequential()
        self.decoder.add_module('fc', nn.Linear(latent_dim, encoder_channels * seed_dims[0] * seed_dims[1]))
        self.decoder.add_module('unflatten', nn.Unflatten(1, (encoder_channels, *seed_dims)))
        for i in range(_DECODER_LAYERS - 1):
            in_filters = filters(_ENCODER_LAYERS - 1 - i)
            out_filters = filters(_ENCODER_LAYERS - 2 - i)
            layer = OrderedDict()
            layer['conv_transposed_2d'] = nn.ConvTranspose2d(
                in_channels=in_filters,
                out_channels=out_filters,
                kernel_size=kernel_size,
                stride=2,
                padding=1,
                output_padding=1,
                bias=not batch_norm,
            )
            if batch_norm:
                layer['batch_norm'] = nn.BatchNorm2d(out_filters)
            layer['leaky_relu'] = nn.LeakyReLU(0.2, inplace=True)

            self.decoder.add_module(f't_conv_layer_{i}', nn.Sequential(layer))
        self.decoder.add_module('conv_transposed_2d', nn.ConvTranspose2d(
            in_channels=filters(_ENCODER_LAYERS - _DECODER_LAYERS),
            out_channels=decoder_output_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
            output_padding=1,
        ))
        self.decoder.add_module('tanh', nn.Tanh())

    def encode(self, x):
        return self.encoder(x)

    def decode(self, latent):
        return self.decoder(latent)

    def forward(self, x):
        return self.decode(self.encode(x))


# name of the architecture in configs and checkpoints written before the rename
AE2 = AutoEncoder


def _conv_out(size, kernel_size, stride=2, padding=1):
    return (size + 2 * padding - kernel_size) // stride + 1


def _grow(size, kernel_size, layers, stride=2, padding=1, output_padding=1):
    for _ in range(layers):
        size = (size - 1) * stride - 2 * padding + kernel_size + output_padding
    return size
