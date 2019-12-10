import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, calculate_gain
import numpy as np


# KLUDGE: This is copied from the audio generation module and should be
# factored out somewhere
class GeneratorBlock(nn.Module):
    """
    Apply a series of increasingly dilated convolutions with optinal upscaling
    at the end
    """

    def __init__(self, dilations, channels, kernel_size, upsample_factor=2):
        super().__init__()
        self.upsample_factor = upsample_factor
        self.kernel_size = kernel_size
        self.channels = channels
        self.dilations = dilations
        layers = []
        for i, d in enumerate(self.dilations):
            padding = (kernel_size * d) // 2
            c = nn.Conv1d(
                channels,
                channels,
                kernel_size,
                dilation=d,
                padding=padding,
                bias=False)
            layers.append(c)

        # Batch norm seems to be responsible for all the annoying high-frequency
        # chirps or blips
        self.main = nn.Sequential(*layers)
        self.bns = nn.Sequential(*[nn.BatchNorm1d(self.channels) for layer in self.main])
        # No overlap helps to avoid buzzing/checkerboard artifacts

        # BE SURE TO SWITCH OUT INITIALIZATION TOO!
        self.activation = lambda x: F.leaky_relu(x, 0.2)

        self.upsampler = nn.ConvTranspose1d(
            channels,
            channels,
            self.upsample_factor * 2,
            stride=self.upsample_factor,
            padding=self.upsample_factor // 2,
            bias=False)

    def forward(self, x):
        batch_size = x.shape[0]
        dim = x.shape[-1]
        x = x.view(x.shape[0], self.channels, -1)

        for i, layer in enumerate(self.main):
            t = layer(x)
            # This residual seems to be very important, at least when using a
            # mostly-positive activation like ReLU
            x = self.activation(x + t[..., :dim])
            # x = self.bns[i](x)

        if self.upsample_factor > 1:
            x = self.upsampler(x)
            # x = F.upsample(x, scale_factor=self.upsample_factor)
            x = self.activation(x)

        return x


class AlternateGenerator(nn.Module):
    def __init__(self, frames, out_channels, noise_dim, initial_dim, channels):
        super().__init__()
        self.initial_dim = initial_dim
        self.out_channels = out_channels
        self.noise_dim = noise_dim
        self.channels = channels
        self.frames = frames

        self.inital = nn.Linear(noise_dim, 8 * 4 * channels)
        n_layers = int(np.log2(frames) - np.log2(8))

        layers = []
        for i in range(n_layers - 1):
            layers.append(nn.ConvTranspose2d(
                channels, channels, (4, 4), (2, 2), (1, 1), bias=False))
        self.main = nn.Sequential(*layers)
        self.to_out_channels = nn.ConvTranspose2d(
            channels, 1, (4, 4), (2, 2), (1, 1), bias=False)


    def initialize_weights(self):
        for name, weight in self.named_parameters():
            if weight.data.dim() > 2:
                if 'to_out_channels' in name:
                    xavier_normal_(weight.data, 1)
                else:
                    xavier_normal_(
                        weight.data, calculate_gain('leaky_relu', 0.2))
        return self

    def forward(self, x):
        x = x.view(-1, self.noise_dim)
        x = F.leaky_relu(self.inital(x), 0.2)
        x = x.view(-1, self.channels, 4, 8)

        for i, layer in enumerate(self.main):
            x = F.leaky_relu(layer(x), 0.2)

        x = self.to_out_channels(x)
        x = x.view(-1, self.out_channels, self.frames)
        return x


class Generator(nn.Module):
    def __init__(self, frames, out_channels, noise_dim, initial_dim, channels):
        super().__init__()
        self.initial_dim = initial_dim
        self.out_channels = out_channels
        self.noise_dim = noise_dim
        self.channels = channels
        self.frames = frames

        self.activation = lambda x: F.leaky_relu(x, 0.2)

        self.to_time_series = nn.Linear(
            self.noise_dim, self.initial_dim * self.channels, bias=False)

        n_layers = int(np.log2(frames) - np.log2(initial_dim))
        layers = []
        for _ in range(n_layers):
            layers.append(
                GeneratorBlock([1, 3, 9], self.channels, 3, upsample_factor=2))
        layers.append(
            GeneratorBlock([1, 3, 9, 1], self.channels, 3, upsample_factor=1))
        self.main = nn.Sequential(*layers)

        self.to_out_channels = nn.Conv1d(
            self.channels, self.out_channels, 1, 1, 0, bias=False)

    def initialize_weights(self):
        for name, weight in self.named_parameters():
            if weight.data.dim() > 2:
                if 'to_out_channels' in name:
                    xavier_normal_(weight.data, 1)
                else:
                    xavier_normal_(
                        weight.data, calculate_gain('leaky_relu', 0.2))
        return self

    def test(self, batch_size):
        inp = torch.FloatTensor(batch_size, self.noise_dim).normal_(0, 1)
        output = self.forward(inp)
        print(output.shape)
        return output

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.noise_dim)
        x = self.activation(self.to_time_series(x))

        x = x.view(batch_size, self.channels, self.initial_dim)
        for layer in self.main:
            x = self.activation(layer(x))

        x = self.to_out_channels(x)
        return F.leaky_relu(x, 0.01)


