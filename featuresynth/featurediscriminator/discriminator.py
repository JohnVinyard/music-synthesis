import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, calculate_gain
import numpy as np


# (kernel_size * d) // 2

class DiscriminatorBlock(nn.Module):
    def __init__(self, dilations, kernel_size, channels, downsample_factor):
        super().__init__()
        self.kernel_size = kernel_size
        self.downsample_factor = downsample_factor
        self.channels = channels
        self.dilations = dilations

        layers = []
        for d in dilations:
            layers.append(nn.Conv1d(
                channels,
                channels,
                self.kernel_size,
                stride=1,
                padding=(kernel_size * d) // 2,
                dilation=d,
                bias=False))
        layers.append(nn.Conv1d(
            channels,
            channels,
            (downsample_factor * 2) + 1,
            stride=downsample_factor,
            padding=downsample_factor,
            bias=False))
        self.main = nn.Sequential(*layers)
        self.activation = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.channels, -1)
        for layer in self.main:
            z = layer(x)
            if z.shape[-1] > x.shape[-1]:
                z = z[..., :x.shape[-1]]
            x = self.activation(z)
        return x


class AlternateDiscriminator(nn.Module):
    def __init__(self, frames, feature_channels, channels, n_judgements):
        super().__init__()
        self.n_judgements = n_judgements
        self.channels = channels
        self.feature_channels = feature_channels
        self.frames = frames
        n_layers = int(np.log2(frames))
        layers = []
        for i in range(n_layers):
            layers.append(nn.Conv2d(1 if i == 0 else channels, channels, (3, 3), (2, 2), (1, 1), bias=False))
        self.main = nn.Sequential(*layers)
        self.judge = nn.Conv2d(channels, 1, (1, 1), (1, 1), bias=False)

    def initialize_weights(self):
        for name, weight in self.named_parameters():
            if weight.data.dim() > 2:
                if 'judge' in name:
                    xavier_normal_(weight.data, calculate_gain('tanh'))
                else:
                    xavier_normal_(
                        weight.data, calculate_gain('leaky_relu', 0.2))
        return self

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, self.feature_channels, self.frames)

        for layer in self.main:
            x = F.leaky_relu(layer(x), 0.2)

        x = self.judge(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, frames, feature_channels, channels, n_judgements):
        super().__init__()
        self.n_judgements = n_judgements
        self.channels = channels
        self.feature_channels = feature_channels
        self.frames = frames

        n_layers = int(np.log2(frames) - np.log2(n_judgements))

        self.embedding = nn.Conv1d(
            feature_channels, self.channels, 1, 1, 0, bias=False)

        layers = []
        for _ in range(n_layers):
            layers.append(DiscriminatorBlock(
                [1, 3, 9], 3, channels, downsample_factor=2))
        self.main = nn.Sequential(*layers)

        self.judgements = nn.Conv1d(channels, 1, 3, 1, 1, bias=False)

        self.activation = lambda x: F.leaky_relu(x, 0.2)

    def initialize_weights(self):
        for name, weight in self.named_parameters():
            if weight.data.dim() > 2:
                if 'judge' in name:
                    xavier_normal_(weight.data, calculate_gain('tanh'))
                else:
                    xavier_normal_(
                        weight.data, calculate_gain('leaky_relu', 0.2))
        return self

    def test(self, batch_size):
        x = torch.FloatTensor(batch_size, self.feature_channels, self.frames)
        features, x = self.forward(x)
        print(features.shape, x.shape)
        return features, x

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.feature_channels, self.frames)

        x = self.activation(self.embedding(x))

        for layer in self.main:
            x = self.activation(layer(x))

        # x = F.tanh(self.judgements(x))
        x = self.judgements(x)
        return x



