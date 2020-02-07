import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, calculate_gain

from .ddsp import oscillator_bank, smooth_upsample2, noise_bank2
from ..util import device
from ..util.modules import DilatedStack
from torch.nn.utils import weight_norm

def weight_norm(x):
    return x

class ResidualAtom(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilation = dilation
        self.channels = channels
        padding = dilation
        self.main = nn.Sequential(
            weight_norm(nn.Conv1d(
                channels,
                channels,
                3,
                1,
                dilation=dilation,
                padding=padding)),
            weight_norm(nn.Conv1d(channels, channels, 3, 1, 1)))

    def forward(self, x):
        orig = x
        for layer in self.main:
            x = F.leaky_relu(layer(x), 0.2)
        return orig + x


class ResidualStack(nn.Module):
    def __init__(self, channels, dilations):
        super().__init__()
        self.dilations = dilations
        self.channels = channels
        self.main = nn.Sequential(
            ResidualAtom(channels, 1),
            ResidualAtom(channels, 3),
            ResidualAtom(channels, 9),
        )

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x


class MelGanGenerator(nn.Module):
    def __init__(self, input_size, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size

        self.main = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels, 512, 7, 1, 3)),
            nn.LeakyReLU(0.2),

            weight_norm(nn.ConvTranspose1d(512, 256, 16, 8, 4)),
            nn.LeakyReLU(0.2),

            ResidualStack(256, [1, 3, 9]),

            weight_norm(nn.ConvTranspose1d(256, 128, 16, 8, 4)),
            nn.LeakyReLU(0.2),

            ResidualStack(128, [1, 3, 9]),

            weight_norm(nn.ConvTranspose1d(128, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2),

            ResidualStack(64, [1, 3, 9]),

            weight_norm(nn.ConvTranspose1d(64, 32, 4, 2, 1)),
            nn.LeakyReLU(0.2),

            ResidualStack(32, [1, 3, 9]),

            weight_norm(nn.Conv1d(32, 1, 7, 1, 3)),
            nn.Tanh()
        )

    def initialize_weights(self):
        for name, weight in self.named_parameters():
            if weight.data.dim() > 2:
                weight.data.normal_(0, 0.02)
        return self

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x


class DDSPGenerator(nn.Module):
    def __init__(self, input_size, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size

        c = 512

        self.main = nn.Sequential(
            nn.Conv1d(in_channels, c, 1, 1, 0, bias=False),
            nn.Conv1d(c, c, 3, 1, 1, bias=False),
            nn.Conv1d(c, c, 3, 1, 1, bias=False),
            nn.Conv1d(c, c, 3, 1, 1, bias=False),
        )

        # self.main = DilatedStack(
        #     in_channels=in_channels,
        #     channels=c,
        #     kernel_size=3,
        #     dilations=[1, 3, 9, 1],
        #     activation=lambda x: F.leaky_relu(x, 0.2),
        #     residual=True)

        total_samples = 16384

        n_osc = 32
        # number of fft coefficients needed for each window of samples
        # total_samples / param sample rate / 2
        self.loudness = nn.Conv1d(c, n_osc, 1, 1, 0, bias=False)
        self.frequency = nn.Conv1d(c, n_osc, 1, 1, 0, bias=False)

        noise_rate = 64
        self.noise_loudness = nn.Conv1d(
            c, (total_samples // noise_rate) + 1, 1, 1, 0, bias=False)

        # n_osc
        stops = np.geomspace(20, 11025 / 2, num=n_osc)
        # n_osc + 1
        freqs = [0] + list(stops)
        # n_osc
        diffs = np.diff(freqs)
        # n_osc
        starts = stops - diffs

        self.starts = torch.from_numpy(starts).to(device).float()
        self.diffs = torch.from_numpy(diffs).to(device).float()

    def initialize_weights(self):
        for name, weight in self.named_parameters():
            if weight.data.dim() > 2:
                if 'samples' in name:
                    xavier_normal_(weight.data, 1)
                else:
                    xavier_normal_(
                        weight.data, calculate_gain('leaky_relu', 0.2))
        return self

    def forward(self, x, debug=False):
        x = x.view(-1, self.in_channels, self.input_size)
        for layer in self.main:
            x = F.leaky_relu(layer(x), 0.2)
        # x = self.main(x)

        # oscillator channel loudness
        l = self.loudness(x) ** 2

        # oscillator channel frequency (constrained within band)
        f = F.sigmoid(self.frequency(x))  # (batch, osc, time)
        f = self.starts[None, :, None] + (f * self.diffs[None, :, None])

        # desired frequency response of FIR filter in the frequency domain
        # x = self.nl(x)
        n_l = self.noise_loudness(x) ** 2

        # l = smooth_upsample2(l, size=16384)
        l = F.upsample(l, size=16384, mode='linear')
        f = F.upsample(f, size=16384, mode='linear')
        # f = smooth_upsample2(f, size=16384)

        harmonic = oscillator_bank(f, l, 11025).view(-1, 1, 16384)
        noise = noise_bank2(n_l)
        if debug:
            return harmonic, noise, l, f, n_l
        else:
            return harmonic
