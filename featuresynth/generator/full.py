import numpy as np
import torch
import zounds
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, calculate_gain
from torch.nn.utils import weight_norm

from .ddsp import oscillator_bank, smooth_upsample2, noise_bank2
from ..util import device
from ..util.modules import normalize, UpsamplingStack, \
    UpSample, ResidualStack


def weight_norm(x):
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
                if 'samples' in name:
                    xavier_normal_(weight.data, 1)
                else:
                    xavier_normal_(
                        weight.data, calculate_gain('leaky_relu', 0.2))
        return self

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x


class TwoDimDDSPGenerator(nn.Module):
    def __init__(self, input_size, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size

        n_osc = 128

        # go from (batch, 1, bins, time) => (batch, 2, n_osc, time)
        self.main = nn.Sequential(
            nn.Conv2d(1, 32, (3, 9), (1, 1), (1, 4)),

            nn.Conv2d(32, 32, (9, 3), (1, 1), (4, 1)),

            nn.Conv2d(32, 64, (3, 9), (1, 1), (1, 4)),
            # downsample step
            nn.Conv2d(64, 64, (3, 9), (2, 1), (1, 4)),
            nn.Conv2d(64, 32, (3, 9), (1, 1), (1, 4)),
        )

        # self.noise = nn.Sequential(
        #
        #     # (batch, 32, 128, 64)
        #     nn.Conv2d(32, 16, (9, 3), (4, 1), (4, 1)),
        #     nn.LeakyReLU(0.2),
        #     # (batch, 32, 32, 64)
        #
        #     nn.Upsample(scale_factor=(1, 4), mode='bilinear'),
        #     # (batch, 16, 32, 256)
        #
        #     nn.Conv2d(16, 16, (5, 3), (2, 1), (2, 1)),
        #     nn.LeakyReLU(0.2),
        #     # (batch, 16, 16, 256)
        #
        #     nn.Upsample(scale_factor=(1, 4), mode='bilinear'),
        #     # (batch, 16, 16, 1024)
        #     nn.Conv2d(16, 17, (16, 3), (1, 1), (0, 1))
        #     # (batch, 17, 1, 1024)
        # )


        self.noise = nn.Sequential(

            # downsample frequency
            # (batch, 32, 128, 64)
            nn.Conv2d(32, 16, (17, 3), (8, 1), (8, 1)),
            nn.LeakyReLU(0.2),
            # (batch, 32, 16, 64)

            # upsample time
            nn.Upsample(scale_factor=(1, 16), mode='bilinear'),
            # (batch, 16, 16, 1024)

            nn.Conv2d(16, 16, (17, 3), (8, 1), (8, 1)),
            nn.LeakyReLU(0.2),
            # (batch, 16, 1, 1024)

            nn.Conv2d(16, 16, (2, 3), (1, 1), (0, 1)),
            nn.LeakyReLU(0.2),

            nn.Conv2d(16, 17, (3, 3), (1, 1), (1, 1)),

        )

        self.to_params = nn.Conv2d(32, 2, (3, 3), (1, 1), (1, 1))

        # self.freqs = nn.Conv2d(32, 1, (3, 3), (1, 1), (1, 1))
        # self.osc_loudness = nn.Conv2d(32, 1, (3, 3), (1, 1), (1, 1))
        # self.noise_loudness = nn.Conv2d(32, 1, (3, 3), (1, 1), (1, 1))

        sr = zounds.SR11025()
        # scale = zounds.MelScale(zounds.FrequencyBand(20, sr.nyquist), n_osc)
        scale = zounds.GeometricScale(20, sr.nyquist - 20, 0.05, n_osc)
        centers = np.array([b.center_frequency for b in scale])
        erbs = (centers * 0.108) + 24.7

        self.centers = torch.from_numpy(centers).to(device).float()
        self.erbs = torch.from_numpy(erbs).to(device).float()

        noise = torch.FloatTensor(1, 1, 24576).uniform_(-1, 1).to(device)
        fb = zounds.learn.FilterBank(
            sr,
            128,
            scale,
            np.linspace(0.25, 0.5, len(scale)),
            normalize_filters=False,
            a_weighting=False).to(device)
        filtered_noise = fb.convolve(noise)[..., :24576]
        self.filtered_noise = filtered_noise

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
                if name in {'to_params', 'freqs', 'osc_loudness',
                            'noise_loudness'}:
                    xavier_normal_(weight.data, 1)
                else:
                    xavier_normal_(
                        weight.data, calculate_gain('leaky_relu', 0.2))
        return self

    # def forward(self, x):
    #     batch, channels, time = x.shape
    #     x = x.view(batch, 1, channels, time)
    #
    #     for layer in self.main:
    #         x = F.leaky_relu(layer(x), 0.2)
    #
    #     x = self.to_params(x)
    #
    #     f = F.tanh(x[:, 1, :, :])
    #     f = self.centers[None, :, None] + (f * self.erbs[None, :, None])
    #
    #     # f = F.sigmoid(x[:, 0, :, :]).view(batch, -1, time)
    #     # f = self.starts[None, :, None] + (f * self.diffs[None, :, None])
    #
    #     osc_l = (x[:, 1, :, :] ** 2).view(batch, -1, time)
    #     n_l = (x[:, 2, :, :] ** 2).view(batch, -1, time)
    #
    #     # f = F.sigmoid(self.freqs(x)).view(batch, -1, time)
    #     # osc_l = (self.osc_loudness(x) ** 2).view(batch, -1, time)
    #     # n_l = (self.noise_loudness(x) ** 2).view(batch, -1, time)
    #
    #     scale_factor = 256
    #     padding = 8192
    #     half_pad = padding // 2
    #
    #
    #     f = smooth_upsample2(f, size=time * scale_factor)
    #     osc_l = smooth_upsample2(osc_l, size=time * scale_factor)
    #     n_l = smooth_upsample2(n_l, size=time * scale_factor)
    #
    #
    #     noise = (n_l * self.filtered_noise).sum(dim=1, keepdim=True)
    #     harmonic = oscillator_bank(f, osc_l, 11025).view(batch, 1, -1)
    #     return (harmonic + noise)[..., half_pad:-half_pad]


    def forward(self, x):
        batch, channels, time = x.shape
        orig = x

        x = x.view(batch, 1, channels, time)

        for layer in self.main:
            x = F.leaky_relu(layer(x), 0.2)

        pre_params = x
        x = self.to_params(x)
        l = x[:, 0, :, :] ** 2

        # f = F.sigmoid(x[:, 1, :, :])
        # f = self.starts[None, :, None] + (f * self.diffs[None, :, None])

        f = F.tanh(x[:, 1, :, :]) * 0.5
        f = self.centers[None, :, None] + (f * self.erbs[None, :, None])

        # l = F.upsample(l, scale_factor=256, mode='linear')
        l = smooth_upsample2(l, size=x.shape[-1] * 256)
        # f = F.upsample(f, scale_factor=256, mode='linear')
        f = smooth_upsample2(f, size=x.shape[-1] * 256)

        # desired frequency response of FIR filter in the frequency domain
        # n_l = self.noise(pre_params) ** 2
        # print(pre_params.shape)
        for layer in self.noise:
            pre_params = layer(pre_params)
            # print(pre_params.shape)
        n_l = pre_params ** 2
        # print(n_l.shape)

        # print(nl.shape, self.filtered_noise.shape)
        # nl = x[:, 2, :, :] ** 2
        # nl = smooth_upsample2(nl, size=x.shape[-1] * 256)
        # noise = (nl * self.filtered_noise).sum(dim=1, keepdim=True)

        harmonic = oscillator_bank(f, l, 11025).view(x.shape[0], 1, -1)
        noise = noise_bank2(n_l.view(batch, 17, -1))

        # print(harmonic.shape, noise.shape)
        return (harmonic + noise)[:, :, 4096:-4096]


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

        n_osc = 128
        self.n_osc = n_osc

        sr = zounds.SR11025()
        # scale = zounds.MelScale(zounds.FrequencyBand(20, sr.nyquist), n_osc)
        scale = zounds.GeometricScale(20, sr.nyquist - 20, 0.05, n_osc)
        centers = np.array([b.center_frequency for b in scale])
        erbs = (centers * 0.108) + 24.7

        self.centers = torch.from_numpy(centers).to(device).float()
        self.erbs = torch.from_numpy(erbs).to(device).float()

        total_samples = 16384
        self.total_samples = total_samples

        # number of fft coefficients needed for each window of samples
        # total_samples / param sample rate / 2

        self.frequency = nn.Conv1d(c, n_osc * 2, 1, 1, 0, bias=False)
        # self.loudness = nn.Conv1d(n_osc + c, n_osc, 1, 1, 0, bias=False)

        noise_rate = 1024
        self.nl = UpsamplingStack(
            start_size=64,
            target_size=1024,
            scale_factor=2,
            layer_func=lambda i, curr_size, out_size, first, last: UpSample(c,
                                                                            c,
                                                                            7,
                                                                            2,
                                                                            activation=lambda
                                                                                x: F.leaky_relu(
                                                                                x,
                                                                                0.2))
        )
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
        x = x.view(x.shape[0], self.in_channels, -1)
        for layer in self.main:
            x = F.leaky_relu(layer(x), 0.2)
        # x = self.main(x)


        osc = self.frequency(x)
        l = osc[:, :self.n_osc, :] ** 2
        # f = F.sigmoid(osc[:, self.n_osc:, :])
        # f = self.starts[None, :, None] + (f * self.diffs[None, :, None])

        f = F.tanh(osc[:, self.n_osc:, :]) * 0.5
        f = self.centers[None, :, None] + (f * self.erbs[None, :, None])

        # nyquist = 11025 / 2
        # f = ((1e-12 + f) ** 2) * nyquist
        # f = f * nyquist

        # desired frequency response of FIR filter in the frequency domain
        x = self.nl(x)
        n_l = self.noise_loudness(x) ** 2

        l = smooth_upsample2(l, size=24576)
        # l = F.upsample(l, scale_factor=256, mode='linear')
        # f = F.upsample(f, scale_factor=256, mode='linear')
        f = smooth_upsample2(f, size=24576)

        harmonic = oscillator_bank(f, l, 11025).view(x.shape[0], 1, -1)
        noise = noise_bank2(n_l)
        # if debug:
        #     return harmonic, noise, l, f, n_l
        # else:
        #     return normalize(harmonic)
        x = normalize(harmonic + noise)[:, :, 4096:-4096]
        return x
