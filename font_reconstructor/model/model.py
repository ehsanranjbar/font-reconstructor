from typing import Optional, Tuple

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base class for all models. A model encodes a text image to a latent vector and decodes that to a fingerprint.
    """

    def encode(self, x):
        raise NotImplementedError

    def decode(self, latent):
        raise NotImplementedError

    def decode_glyphs(self, latent, glyph_index):
        """
        Draw only some glyphs of each fingerprint.

        :param glyph_index: tensor (batch, k) of the glyphs to draw for each latent vector
        :return: tensor (batch, k, height, width)
        """
        return select_glyphs(self.decode(latent), glyph_index)

    def forward(self, x, return_latent=False, glyphs=None):
        """
        Forward pass logic

        :param return_latent: also return the latent vector, which losses on the embedding need
        :param glyphs: tensor (batch, k) of glyph indices. If given, only these glyphs of each fingerprint are
            drawn, see `decode_glyphs`.
        :return: the reconstructed fingerprint, or (fingerprint, latent) with `return_latent`
        """
        latent = self.encode(x)
        output = self.decode(latent) if glyphs is None else self.decode_glyphs(latent, glyphs)
        return (output, latent) if return_latent else output

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


def select_glyphs(fingerprint, glyph_index):
    """
    :param fingerprint: tensor (batch, glyphs, height, width)
    :param glyph_index: tensor (batch, k) of glyphs to pick from each fingerprint
    :return: tensor (batch, k, height, width)
    """
    index = glyph_index[:, :, None, None].expand(-1, -1, *fingerprint.shape[2:])
    return fingerprint.gather(1, index)


class ResidualBlock(nn.Module):
    """
    two 3x3 convolutions with a shortcut, the first one strided to downsample
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.activation(self.body(x) + self.shortcut(x))


class UpsampleBlock(nn.Module):
    """
    Double the resolution by nearest neighbour resizing followed by a convolution.

    Unlike a strided transposed convolution, every output pixel gets the same number of contributions,
    which avoids checkerboard artifacts.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CompactAutoEncoder(BaseModel):
    """
    Small residual autoencoder from a text image to the fingerprint of its font.

    The encoder is a stack of residual blocks that each halve the image, followed by global average pooling
    and a projection to the latent vector. Pooling makes the latent independent of where a stroke sits in the
    image and of the image width, so texts of any length can be encoded.

    The decoder grows a small feature map to the fingerprint with resize-convolution blocks.

    Only the encoder is needed to identify a font, the decoder is a training aid. That is why the two are
    sized separately: `base_channels` and `latent_dim` set the cost of using the model, while
    `decoder_channels` and `decoder_blocks` only make training slower.

    Two kinds of decoder are available with `decoder_type`:

    'joint'        draws all glyphs at once, one output channel per glyph.
    'conditioned'  draws one glyph at a time from the latent vector and a learned embedding of the character.
                   All glyphs share its weights, so what it learns about a style applies to every character.
                   It runs once per glyph, which makes training slower than with the joint decoder.

    :param base_channels: channels of the first encoder stage, doubled by each following stage
    :param latent_dim: size of the latent vector
    :param encoder_stages: number of downsampling residual blocks
    :param decoder_output_channels: number of glyphs in the font fingerprint
    :param input_dims: (height, width) of the text image. Only its minimum size matters.
    :param output_dims: (height, width) of each glyph of the fingerprint, multiples of 16
    :param decoder_channels: base width of the decoder, `base_channels` if None
    :param decoder_blocks: convolution blocks at each resolution of the decoder. The first one upsamples, the
        others are residual blocks.
    :param decoder_type: 'joint' or 'conditioned'
    :param glyph_embedding_dim: size of the character embedding of the conditioned decoder
    :param style_classes: number of font styles. If given, a linear layer on the latent vector predicts the
        style of the font, see `predict_style`.
    """

    _UPSAMPLINGS = 4
    # glyphs the conditioned decoder draws in one pass. Drawing all glyphs of a large batch at once would hold
    # every one of them in memory at full resolution.
    _GLYPHS_PER_PASS = 4096

    def __init__(
            self,
            base_channels: int = 8,
            latent_dim: int = 32,
            encoder_stages: int = 4,
            decoder_output_channels: int = 42,
            input_dims: Tuple[int, int] = (32, 128),
            output_dims: Tuple[int, int] = (32, 32),
            decoder_channels: Optional[int] = None,
            decoder_blocks: int = 1,
            decoder_type: str = 'joint',
            glyph_embedding_dim: int = 16,
            style_classes: int = 0,
    ):
        super().__init__()
        if decoder_type not in ('joint', 'conditioned'):
            raise ValueError(f"Unknown decoder_type '{decoder_type}'. Valid options are 'joint' and 'conditioned'.")
        if decoder_blocks < 1:
            raise ValueError("decoder_blocks has to be at least 1.")
        self.decoder_type = decoder_type
        self.latent_dim = latent_dim
        self.input_dims = tuple(input_dims)
        self.output_dims = tuple(output_dims)
        self.output_channels = decoder_output_channels

        if min(self.input_dims) < 2 ** encoder_stages:
            raise ValueError(f"input_dims {self.input_dims} are too small for {encoder_stages} encoder stages.")
        scale = 2 ** self._UPSAMPLINGS
        if any(size < scale or size % scale for size in self.output_dims):
            raise ValueError(f"output_dims {self.output_dims} have to be multiples of {scale}.")

        # encoder
        channels = [base_channels * 2 ** i for i in range(encoder_stages)]
        stages = [
            nn.Conv2d(1, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        in_channels = channels[0]
        for out_channels in channels:
            stages.append(ResidualBlock(in_channels, out_channels, stride=2))
            in_channels = out_channels
        self.encoder = nn.Sequential(*stages)
        self.to_latent = nn.Linear(channels[-1], latent_dim)
        self.style_head = nn.Linear(latent_dim, style_classes) if style_classes else None

        # decoder
        width = base_channels if decoder_channels is None else decoder_channels
        widths = [width * 8, width * 4, width * 2, width * 2, width * 2]
        seed_dims = (self.output_dims[0] // scale, self.output_dims[1] // scale)

        conditioned = decoder_type == 'conditioned'
        if conditioned:
            self.glyph_embedding = nn.Embedding(decoder_output_channels, glyph_embedding_dim)
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim + (glyph_embedding_dim if conditioned else 0),
                      widths[0] * seed_dims[0] * seed_dims[1]),
            nn.Unflatten(1, (widths[0], *seed_dims)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        layers = []
        for i in range(self._UPSAMPLINGS):
            layers.append(UpsampleBlock(widths[i], widths[i + 1]))
            layers.extend(ResidualBlock(widths[i + 1], widths[i + 1]) for _ in range(decoder_blocks - 1))
        self.decoder = nn.Sequential(
            *layers,
            nn.Conv2d(widths[-1], 1 if conditioned else decoder_output_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def encode(self, x):
        features = self.encoder(x)
        return self.to_latent(features.mean(dim=(2, 3)))

    def predict_style(self, latent):
        """
        :return: tensor (batch, style_classes) of scores, the highest one is the predicted style
        """
        if self.style_head is None:
            raise ValueError("The model has no style head, build it with style_classes.")
        return self.style_head(latent)

    def decode(self, latent):
        if self.decoder_type == 'joint':
            return self.decoder(self.from_latent(latent))

        # one pass per glyph: every latent vector is paired with every character embedding
        batch_size, glyphs = latent.shape[0], self.output_channels
        styles = latent.unsqueeze(1).expand(batch_size, glyphs, -1)
        characters = self.glyph_embedding.weight.unsqueeze(0).expand(batch_size, glyphs, -1)
        conditions = torch.cat([styles, characters], dim=2).reshape(batch_size * glyphs, -1)
        return self._draw(conditions).reshape(batch_size, glyphs, *self.output_dims)

    def decode_glyphs(self, latent, glyph_index):
        """
        Draw only some glyphs of each fingerprint.

        The conditioned decoder runs for these glyphs only, which is what makes training on a few glyphs of
        each sample affordable. The joint decoder draws all glyphs anyway, they are picked from its output.

        :param glyph_index: tensor (batch, k) of the glyphs to draw for each latent vector
        :return: tensor (batch, k, height, width)
        """
        if self.decoder_type == 'joint':
            return super().decode_glyphs(latent, glyph_index)
        batch_size, glyphs = glyph_index.shape
        styles = latent.unsqueeze(1).expand(batch_size, glyphs, -1)
        conditions = torch.cat([styles, self.glyph_embedding(glyph_index)], dim=2).reshape(batch_size * glyphs, -1)
        return self._draw(conditions).reshape(batch_size, glyphs, *self.output_dims)

    def _draw(self, conditions):
        """
        the conditioned decoder on rows of (latent vector, character embedding)

        :return: tensor (rows, height, width)
        """
        images = [self.decoder(self.from_latent(chunk)) for chunk in conditions.split(self._GLYPHS_PER_PASS)]
        if not images:
            return conditions.new_zeros(0, *self.output_dims)
        return torch.cat(images).squeeze(1)


class GlyphDiscriminator(nn.Module):
    """
    Judges whether a font fingerprint is real or drawn by the decoder, patch by patch.

    It looks at the glyphs of a fingerprint together, one input channel per glyph as in MC-GAN (Azadi et al.,
    2018), and outputs a grid of scores. Each score judges a patch of the fingerprint, which pushes the decoder
    towards sharp, plausible strokes everywhere instead of the blurry average that a pixel loss settles for.

    :param in_channels: number of glyphs in the fingerprint
    :param base_channels: channels of the first convolution
    """

    def __init__(self, in_channels: int = 42, base_channels: int = 32):
        super().__init__()

        def conv(cin, cout, stride):
            # spectral normalization keeps the discriminator from overpowering the decoder
            return nn.utils.spectral_norm(nn.Conv2d(cin, cout, kernel_size=4, stride=stride, padding=1))

        self.layers = nn.Sequential(
            conv(in_channels, base_channels, 2),
            nn.LeakyReLU(0.2, inplace=True),
            conv(base_channels, base_channels * 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            conv(base_channels * 2, base_channels * 4, 1),
            nn.LeakyReLU(0.2, inplace=True),
            conv(base_channels * 4, 1, 1),
        )

    def forward(self, fingerprint):
        return self.layers(fingerprint)
