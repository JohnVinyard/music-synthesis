import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, calculate_gain
from ..util.modules import \
    DilatedStack, normalize, flatten_channels, unflatten_channels, \
    UpsamplingStack, LearnedUpSample, ToTimeSeries
import numpy as np


class EnergyGenerator(nn.Module):
    """
    This generator creates loudness time series
    """

    def __init__(self, frames, out_channels, noise_dim, initial_dim, channels,
                 ae):
        super().__init__()
        self.ae = [ae]
        self.initial_dim = initial_dim
        self.out_channels = out_channels
        self.noise_dim = noise_dim
        self.channels = channels
        self.frames = frames

        self.to_time_series = nn.Linear(
            noise_dim, initial_dim * channels, bias=False)

        self.initial = nn.Sequential(
            nn.Conv1d(channels, channels, 3, 1, dilation=1, padding=1,
                      bias=False),
            nn.Conv1d(channels, channels, 3, 1, dilation=3, padding=3,
                      bias=False),
        )

        c = channels

        self.loudness = nn.Sequential(
            nn.Conv1d(c, c, 3, 1, dilation=27, padding=27, bias=False),
            nn.Conv1d(c, c, 3, 1, dilation=9, padding=9, bias=False),
            nn.Conv1d(c, c, 3, 1, dilation=3, padding=3, bias=False),
            nn.Conv1d(c, c, 3, 1, dilation=1, padding=1, bias=False),
            nn.Conv1d(c, c, 3, 1, dilation=1, padding=1, bias=False),
            nn.Conv1d(c, c, 3, 1, dilation=1, padding=1, bias=False),
        )

        self.to_frames = nn.Conv1d(channels, 1, 7, 1, 3, bias=False)



    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = F.leaky_relu(self.to_time_series(x), 0.2)
        x = x.view(batch_size, self.channels, self.initial_dim)

        for layer in self.initial:
            z = layer(x)
            x = F.leaky_relu(x + z, 0.2)

        x = F.upsample(x, size=self.frames)

        for layer in self.loudness:
            z = layer(x)
            x = F.leaky_relu(x + z, 0.2)

        x = F.sigmoid(self.to_frames(x))
        return x


class GeneratorBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            channels,
            noise_dim,
            factor,
            conv_spec_kernel_size=0,
            initial_size=8):

        super().__init__()
        self.initial_size = initial_size
        self.conv_spec_kernel_size = conv_spec_kernel_size
        self.factor = factor
        self.noise_dim = noise_dim
        self.channels = channels
        self.in_channels = in_channels

        self.out_channels = in_channels * factor

        self.embedding = nn.Conv1d(in_channels, channels, 1, 1, 0, bias=False)


        self.stack = DilatedStack(
            channels,
            channels,
            3,
            [27, 9, 3, 1, 1, 1],
            activation=lambda x: F.leaky_relu(x, 0.2),
            residual=True)
        if self.conv_spec_kernel_size > 0:
            self.to_spec_series = ToTimeSeries(
                channels, channels, initial_size)
            self.spec_stack = UpsamplingStack(
                self.initial_size,
                self.out_channels,
                scale_factor=2,
                layer_func=self._build_spec_layer)
        else:
            self.to_frames = nn.Conv1d(
                channels, self.out_channels, 1, 1, 0, bias=False)

    def _build_spec_layer(self, i, curr_size, out_size, first, last):
        return LearnedUpSample(
            in_channels=self.channels,
            out_channels=1 if last else self.channels,
            kernel_size=self.conv_spec_kernel_size,
            scale_factor=2,
            activation=self._to_frames if last else self._relu)

    def _relu(self, x):
        return F.leaky_relu(x, 0.2)

    def _to_frames(self, x):
        return normalize(x ** 2)

    def forward(self, noise, conditioning):
        batch_size = conditioning.shape[0]
        c = conditioning.view(batch_size, self.in_channels, -1)
        # n = noise.view(batch_size, self.noise_dim, 1).repeat(1, 1, c.shape[-1])
        embedded = self.embedding(c)
        # x = torch.cat([c, embedded], dim=1)
        x = embedded
        x = self.stack(x)

        if self.conv_spec_kernel_size > 0:
            x = flatten_channels(x)
            x = self.to_spec_series(x)
            x = self.spec_stack(x)
            x = unflatten_channels(x, batch_size)
        else:
            x = normalize(self.to_frames(x) ** 2)

        # force loudness to match conditioning
        loudness = normalize(conditioning.sum(dim=1, keepdim=True))
        mx, _ = x.max(dim=1, keepdim=True)
        x = (x / (mx + 1e-12)) * loudness
        return x


class Generator(nn.Module):
    def __init__(
            self, frames, out_channels, noise_dim, initial_dim, channels, ae):
        super().__init__()
        self.ae = ae
        self.channels = channels
        self.initial_dim = initial_dim
        self.noise_dim = noise_dim
        self.out_channels = out_channels
        self.frames = frames

        self.loudness = EnergyGenerator(
            frames, out_channels, noise_dim, initial_dim, channels, ae)
        self.four = GeneratorBlock(1, channels, noise_dim, 4)
        self.sixteen = GeneratorBlock(4, channels, noise_dim, 4)
        self.sixty_four = GeneratorBlock(16, channels, noise_dim, 4)
        self.two_fifty_six = GeneratorBlock(64, channels, noise_dim, 4)

        self.layers = {
            1: self.loudness,
            4: self.four,
            16: self.sixteen,
            64: self.sixty_four,
            256: self.two_fifty_six
        }

    def initialize_weights(self):
        for name, weight in self.named_parameters():
            if weight.data.dim() > 2:
                if 'to_frames' in name:
                    xavier_normal_(weight.data, 1)
                else:
                    xavier_normal_(weight.data,
                                   calculate_gain('leaky_relu', 0.2))
        return self

    def forward(self, x, output_size=None):
        noise = x
        x = self.loudness(noise)
        for size, layer in self.layers.items():
            if size > 1:
                x = layer(noise, x)
            if size == output_size:
                return x
        return x

def make_stack(start_size, target_size, layer_func):
    n_layers = int(np.log2(target_size) - np.log2(start_size))
    layers = []
    for i in range(n_layers):
        layers.append(layer_func(i))
    return nn.Sequential(*layers)

class LowResGenerator(nn.Module):
    def __init__(self, out_channels, noise_dim):
        super().__init__()
        self.noise_dim = noise_dim
        self.out_channels = out_channels

        self.initial = nn.Linear(noise_dim, 4 * 4 * 1024, bias=False)
        self.stack = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, (4, 4), (2, 2), (1, 1), bias=False),
            # (8, 8)
            nn.ConvTranspose2d(512, 256, (4, 4), (2, 2), (1, 1), bias=False),
            # (16, 16)
            nn.ConvTranspose2d(256, 128, (4, 4), (2, 2), (1, 1), bias=False),
            # (32, 32)
            nn.ConvTranspose2d(128, 128, (4, 4), (2, 2), (1, 1), bias=False),
            # (64, 64)
            nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), (1, 1), bias=False),
            # (128, 128)
            nn.ConvTranspose2d(64, 32, (3, 4), (1, 2), (1, 1), bias=False),
            # (128, 256)
            nn.ConvTranspose2d(32, 1, (3, 4), (1, 2), (1, 1), bias=False),
            # (128, 512)
        )

    def forward(self, x):
        x = x.view(-1, self.noise_dim)
        x = F.leaky_relu(self.initial(x), 0.2)
        x = x.view(x.shape[0], -1, 4, 4)
        for i, layer in enumerate(self.stack):
            if i == len(self.stack) - 1:
                x = layer(x)
            else:
                x = F.leaky_relu(layer(x), 0.2)

        x = x.view(x.shape[0], self.out_channels, -1)
        return x
