from typing import OrderedDict

import torch.nn as nn

from base import BaseModel


class AE(BaseModel):
    def __init__(
            self,
            kernel_size=3,
            latent_dim=128,
            decoder_output_channels=42,
    ):
        super().__init__()

        self.encoder = nn.Sequential(OrderedDict([
            ('conv2d_1', nn.Conv2d(1, 16, kernel_size=kernel_size, stride=2, padding=1)),
            ('leaky_relu_1', nn.LeakyReLU(0.2, inplace=True)),
            ('conv2d_2', nn.Conv2d(16, 32, kernel_size=kernel_size, stride=2, padding=1)),
            ('leaky_relu_2', nn.LeakyReLU(0.2, inplace=True)),
            ('conv2d_3', nn.Conv2d(32, 64, kernel_size=kernel_size, stride=2, padding=1)),
            ('leaky_relu_3', nn.LeakyReLU(0.2, inplace=True)),
            ('conv2d_4', nn.Conv2d(64, 128, kernel_size=kernel_size, stride=2, padding=1)),
            ('leaky_relu_4', nn.LeakyReLU(0.2, inplace=True)),
            ('conv2d_5', nn.Conv2d(128, 256, kernel_size=kernel_size, stride=2, padding=1)),
            ('leaky_relu_5', nn.LeakyReLU(0.2, inplace=True)),
            ('conv2d_6', nn.Conv2d(256, 512, kernel_size=kernel_size, stride=2, padding=1)),
            ('leaky_relu_6', nn.LeakyReLU(0.2, inplace=True)),
            ('flatten', nn.Flatten()),
            ('fc', nn.Linear(256 * 1 * 4, latent_dim))
        ]))

        self.decoder = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(latent_dim, 512 * 1 * 4)),
            ('unflatten', nn.Unflatten(1, (512, 2, 2))),
            ('t_conv2d_1', nn.ConvTranspose2d(512, 256, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)),
            ('leaky_relu_1', nn.LeakyReLU(0.2, inplace=True)),
            ('t_conv2d_2', nn.ConvTranspose2d(256, 128, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)),
            ('leaky_relu_2', nn.LeakyReLU(0.2, inplace=True)),
            ('t_conv2d_3', nn.ConvTranspose2d(128, 64, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)),
            ('leaky_relu_3', nn.LeakyReLU(0.2, inplace=True)),
            ('t_conv2d_4', nn.ConvTranspose2d(64, decoder_output_channels, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)),
            ('tanh', nn.Tanh())
        ]))

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AE2(BaseModel):
    def __init__(
            self,
            kernel_size=3,
            base_conv_filters=16,
            batch_norm=False,
            latent_dim=128,
            decoder_output_channels=42,
    ):
        super().__init__()

        self.encoder = nn.Sequential()
        for i in range(6):
            layer = OrderedDict()
            layer['conv2d'] = nn.Conv2d(
                in_channels=1 if i == 0 else base_conv_filters * 2 ** (i - 1),
                out_channels=base_conv_filters * 2 ** i,
                kernel_size=kernel_size,
                stride=2,
                padding=1,
                bias=not batch_norm,
            )
            if batch_norm:
                layer['batch_norm'] = nn.BatchNorm2d(base_conv_filters * 2 ** i)
            layer['leaky_relu'] = nn.LeakyReLU(0.2, inplace=True)

            self.encoder.add_module(f'conv_layer_{i}', nn.Sequential(layer))
        self.encoder.add_module('flatten', nn.Flatten())
        self.encoder.add_module('fc', nn.Linear(256 * 1 * 4, latent_dim))

        self.decoder = nn.Sequential()
        self.decoder.add_module('fc', nn.Linear(latent_dim, 512 * 1 * 4))
        self.decoder.add_module('unflatten', nn.Unflatten(1, (512, 2, 2)))
        for i in range(3):
            layer = OrderedDict()
            layer['conv_transposed_2d'] = nn.ConvTranspose2d(
                in_channels=base_conv_filters * 2 ** (5 - i),
                out_channels=base_conv_filters * 2 ** (4 - i),
                kernel_size=kernel_size,
                stride=2,
                padding=1,
                output_padding=1,
                bias=not batch_norm,
            )
            if batch_norm:
                layer['batch_norm'] = nn.BatchNorm2d(base_conv_filters * 2 ** (4 - i))
            layer['leaky_relu'] = nn.LeakyReLU(0.2, inplace=True)

            self.decoder.add_module(f't_conv_layer_{i}', nn.Sequential(layer))
        self.decoder.add_module('conv_transposed_2d', nn.ConvTranspose2d(
            in_channels=base_conv_filters * 4,
            out_channels=decoder_output_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
            output_padding=1,
        ))
        self.decoder.add_module('tanh', nn.Tanh())

    def forward(self, x):
        return self.decoder(self.encoder(x))
