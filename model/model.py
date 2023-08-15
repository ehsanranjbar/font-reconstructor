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