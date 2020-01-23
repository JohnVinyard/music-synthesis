from torch import nn
from torch.nn import functional as F
import torch
import math


def flatten_channels(x):
    batch, channels, frames = x.shape
    x = x.permute(0, 2, 1).contiguous()
    # TODO: I'm not sure this is correct!
    x = x.view(batch * frames, channels, 1)


def unflatten_channels(x, batch_size):
    # we start with (batch * frames, 1, channels)
    channels = x.shape[-1]
    x = x.view(batch_size, -1, channels)
    x = x.permute(0, 2, 1).contiguous()
    # now we have (batch, channels, frames)
    return x


def normalize(l, scaling=1):
    batch_size = l.shape[0]
    orig_shape = l.shape
    l = l.view(batch_size, -1)
    mx, _ = torch.abs(l).max(dim=1, keepdim=True)
    l = l / (mx + 1e-8)
    l = l.view(*orig_shape)
    return l * scaling


class ToTimeSeries(nn.Module):
    def __init__(self, in_channels, out_channels, out_frames):
        super().__init__()
        self.out_frames = out_frames
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.linear = nn.Linear(
            in_channels, out_channels * out_frames, bias=False)

    def forward(self, x):
        x = x.view(-1, self.in_channels)
        x = self.linear(x)
        x = x.view(-1, self.out_channels, self.out_frames)
        return x


class DilatedStack(nn.Module):
    def __init__(
            self,
            in_channels,
            channels,
            kernel_size,
            dilations,
            activation,
            residual=True):

        super().__init__()
        self.residual = residual
        self.activation = activation
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.channels = channels
        self.in_channels = in_channels

        layers = []
        for i, d in enumerate(dilations):
            layers.append(nn.Conv1d(
                in_channels if i == 0 else channels,
                channels,
                kernel_size,
                padding=d,
                dilation=d,
                bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.shape[0], self.in_channels, -1)
        for layer in self.main:
            if self.residual:
                x = self.activation(layer(x) + x)
            else:
                x = self.activation(layer(x))
        return x


class UpSample(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            scale_factor,
            activation):
        super().__init__()
        self.activation = activation
        self.scale_factor = scale_factor
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=False)

    def forward(self, x):
        x = F.upsample(x, scale_factor=self.scale_factor)
        x = self.activation(self.conv(x))
        return x


class LearnedUpSample(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            scale_factor,
            activation):
        super().__init__()
        self.activation = activation
        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=scale_factor,
            padding=(kernel_size // 2) - 1,
            bias=False)

    def forward(self, x):
        return self.activation(self.conv(x))


class UpsamplingStack(nn.Module):
    def __init__(
            self,
            start_size,
            target_size,
            scale_factor,
            layer_func):

        super().__init__()
        self.layer_func = layer_func
        self.start_size = start_size
        self.target_size = target_size
        self.scale_factor = scale_factor
        n_layers = int(
            math.log(target_size, scale_factor) -
            math.log(start_size, scale_factor))
        layers = []
        curr_size = start_size
        for i in range(n_layers):
            first = i == 0
            last = i == n_layers - 1
            out_size = curr_size * scale_factor
            layers.append(layer_func(i, curr_size, out_size, first, last))
            curr_size = out_size
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x


class DownsamplingStack(nn.Module):
    def __init__(
            self,
            start_size,
            target_size,
            scale_factor,
            layer_func,
            activation):

        super().__init__()
        self.activation = activation
        self.scale_factor = scale_factor
        self.target_size = target_size
        self.start_size = start_size

        n_layers = int(
            math.log(start_size, scale_factor)
            - math.log(target_size, scale_factor))
        layers = []
        curr_size = start_size
        for i in range(n_layers):
            first = i == 0
            last = i == n_layers - 1
            out_size = curr_size // scale_factor
            layers.append(layer_func(i, curr_size, out_size, first, last))
            curr_size = out_size
        self.main = nn.Sequential(*layers)


    def forward(self, x):
        for layer in self.main:
            x = self.activation(layer(x))
        return x
