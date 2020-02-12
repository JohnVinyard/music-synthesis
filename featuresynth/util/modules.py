from torch import nn
from torch.nn import functional as F
import torch
import math


def flatten_channels(x, channels_as_time=False):
    batch, channels, frames = x.shape
    x = x.permute(0, 2, 1).contiguous()
    if channels_as_time:
        x = x.view(batch * frames, 1, channels)
    else:
        x = x.view(batch * frames, channels, 1)
    return x


def unflatten_channels(x, batch_size):
    # we start with (batch * frames, 1, channels)
    channels = x.squeeze().shape[-1]
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


def least_squares_generator_loss(j):
    return 0.5 * ((j - 1) ** 2).mean()


def hinge_generator_loss(j):
    return (-j).mea()


def least_squares_disc_loss(r_j, f_j):
    return 0.5 * (((r_j - 1) ** 2).mean() + (f_j ** 2).mean())


def hinge_discriminator_loss(r_j, f_j):
    return (F.relu(1 - r_j) + F.relu(1 + f_j)).mean()


def zero_grad(*optims):
    for optim in optims:
        optim.zero_grad()


def set_requires_grad(x, requires_grad):
    if isinstance(x, nn.Module):
        x = [x]
    for item in x:
        for p in item.parameters():
            p.requires_grad = requires_grad


def freeze(x):
    set_requires_grad(x, False)


def unfreeze(x):
    set_requires_grad(x, True)


def noise(n_examples, noise_dim, device):
    return torch.FloatTensor(n_examples, noise_dim).normal_(0, 1).to(device)


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
            residual=True,
            groups=None):

        super().__init__()
        if groups is None:
            groups = [1] * len(dilations)

        self.groups = groups
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
                groups=groups[i],
                bias=False))
        self.main = nn.Sequential(*layers)

    def __iter__(self):
        yield from self.main

    def forward(self, x, return_features=False):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.in_channels, -1)
        features = []
        for layer in self.main:
            z = layer(x)
            if self.residual and z.shape[1] == x.shape[1]:
                x = self.activation(z + x)
            else:
                x = self.activation(z)
            features.append(x)

        if return_features:
            return features, x
        else:
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
            padding=(kernel_size - scale_factor) // 2,
            bias=False)

    def forward(self, x):
        x = self.conv(x)
        return self.activation(x)


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

    def __iter__(self):
        yield from self.main

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

    def __iter__(self):
        yield from self.main

    def forward(self, x):
        for layer in self.main:
            x = self.activation(layer(x))
        return x
