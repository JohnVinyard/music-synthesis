import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

from .ddsp import oscillator_bank, smooth_upsample2, noise_bank2
from ..util import device
from ..util.modules import UpsamplingStack, UpSample, ResidualStack


def weight_norm(x):
    return x


class MelGanGenerator(nn.Module):
    def __init__(self, input_size, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size

        self.main = nn.Sequential(
            nn.ReflectionPad1d(3),
            weight_norm(nn.Conv1d(in_channels, 512, 7, 1, padding=0)),
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

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x






class DDSPGenerator(nn.Module):
    def __init__(
            self,
            n_osc,
            input_size,
            in_channels,
            output_size,
            scale,
            samplerate):

        super().__init__()
        self.samplerate = samplerate
        self.output_size = output_size
        self.in_channels = in_channels
        self.input_size = input_size
        self.scale = scale
        self.upsample_factor = output_size // input_size

        c = 512

        self.main = nn.Sequential(
            nn.Conv1d(in_channels, c, 3, 1, 1),
            nn.Conv1d(c, c, 3, 1, 1),
            nn.Conv1d(c, c, 3, 1, 1),
            nn.Conv1d(c, c, 3, 1, 1),
        )
        self.n_osc = n_osc
        self.frequency = nn.Conv1d(c, n_osc, 3, 1, 1)

        self.main2 =  nn.Sequential(
            nn.Conv1d(in_channels, c, 3, 1, 1),
            nn.Conv1d(c, c, 3, 1, 1),
            nn.Conv1d(c, c, 3, 1, 1),
            nn.Conv1d(c, c, 3, 1, 1),
        )
        self.loudness = nn.Conv1d(c, n_osc, 3, 1, 1)


        centers = np.array([b.center_frequency for b in scale])
        erbs = (centers * 0.108) + 24.7

        self.centers = torch.from_numpy(centers).to(device).float()
        self.erbs = torch.from_numpy(erbs).to(device).float()

        self.total_samples = output_size




        noise_rate = 1024
        self.nl = UpsamplingStack(
            start_size=input_size,
            target_size=1024,
            scale_factor=2,
            layer_func=lambda i, curr_size, out_size, first, last: UpSample(
                in_channels if first else c,
                c,
                7,
                2,
                activation=lambda x: F.leaky_relu(x, 0.2))
        )
        self.noise_loudness = nn.Conv1d(
            c, (self.total_samples // noise_rate) + 1, 3, 1, 1)

        # TODO: derive stops and starts from scale
        # n_osc
        stops = np.geomspace(20, samplerate.nyquist, num=n_osc)
        # n_osc + 1
        freqs = [0] + list(stops)
        # n_osc
        diffs = np.diff(freqs)
        # n_osc
        starts = stops - diffs

        self.starts = torch.from_numpy(starts).to(device).float()
        self.diffs = torch.from_numpy(diffs).to(device).float()

    def _freq(self, x):
        for layer in self.main:
            x = F.leaky_relu(layer(x), 0.2)
        osc = self.frequency(x)
        f = F.sigmoid(osc)
        f = self.starts[None, :, None] + (f * self.diffs[None, :, None])
        return f

    def _loud(self, x):
        for layer in self.main2:
            x = F.leaky_relu(layer(x), 0.2)
        x = self.loudness(x) ** 2
        return x



    def forward(self, x):
        input_size = x.shape[-1]

        x = x.view(x.shape[0], self.in_channels, -1)
        # for layer in self.main:
        #     x = F.leaky_relu(layer(x), 0.2)
        #
        #
        # osc = self.frequency(x)
        # # l = osc[:, :self.n_osc, :] ** 2
        #
        # f = F.sigmoid(osc)
        # f = self.starts[None, :, None] + (f * self.diffs[None, :, None])
        #

        l = self._loud(x)
        f = self._freq(x)

        # f = F.tanh(osc[:, self.n_osc:, :]) * 0.5
        # f = self.centers[None, :, None] + (f * self.erbs[None, :, None])


        # desired frequency response of FIR filter in the frequency domain
        x = self.nl(x)
        n_l = self.noise_loudness(x) ** 2

        # l = smooth_upsample2(l, size=input_size * self.upsample_factor)
        # f = smooth_upsample2(f, size=input_size * self.upsample_factor)
        l = F.upsample(l, scale_factor=self.upsample_factor, mode='linear')
        f = F.upsample(f, scale_factor=self.upsample_factor, mode='linear')

        harmonic = oscillator_bank(f, l, int(self.samplerate)).view(x.shape[0], 1, -1)
        noise = noise_bank2(n_l)
        x = harmonic + noise
        return x