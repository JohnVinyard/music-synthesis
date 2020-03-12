from torch import nn
from torch.nn import functional as F
from ..util.modules import UpsamplingStack, LearnedUpSample, ResidualStack
from torch.nn.utils import weight_norm
import torch


class ResidualStackFilterBankGenerator(nn.Module):
    def __init__(
            self,
            filter_bank,
            in_size,
            out_size,
            in_channels,
            add_weight_norm=True):

        super().__init__()
        self.filter_bank = filter_bank
        self.in_channels = in_channels
        self.out_size = out_size
        self.in_size = in_size
        self.add_weight_norm = add_weight_norm



        self.main = nn.Sequential(
            # nn.Conv1d(256, 512, 7, 1, 3),
            self._conv_layer(in_channels, 512, 7, 1, 3),
            nn.LeakyReLU(0.2),

            # nn.ConvTranspose1d(512, 256, 16, 8, 4),
            self._conv_layer(512, 256, 16, 8, 4),
            nn.LeakyReLU(0.2),

            ResidualStack(256, [1, 3, 9], add_weight_norm),

            # nn.ConvTranspose1d(256, 256, 16, 8, 4),
            self._conv_layer(256, 256, 16, 8, 4),
            nn.LeakyReLU(0.2),

            ResidualStack(256, [1, 3, 9], add_weight_norm),

            # nn.ConvTranspose1d(256, 256, 4, 2, 1),
            self._conv_layer(256, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),

            ResidualStack(256, [1, 3, 9], add_weight_norm),

            # nn.ConvTranspose1d(128, 128, 4, 2, 1),
            self._conv_layer(256, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),

            ResidualStack(256, [1, 3, 9], add_weight_norm),
        )
        # self.to_frames = nn.Conv1d(256, 128, 7, 1, 3)
        self.to_frames = self._conv_layer(256, 128, 7, 1, 3)
        self.to_noise = self._conv_layer(256, 128, 7, 1, 3)

        # noise = torch.normal(0, 1, (1, 1, out_size))
        # filtered_noise = self.filter_bank.convolve(noise)
        # self.register_buffer('filtered_noise', filtered_noise)

    def _conv_layer(self, *args, **kwargs):
        conv = nn.ConvTranspose1d(*args, **kwargs)
        if self.add_weight_norm:
            conv = weight_norm(conv)
        return conv


    def forward(self, x):


        x = self.main(x)

        batch, channels, time = x.shape

        noise = self.to_noise(x)

        raw_noise = torch.normal(0, 1, (1, 1, time)).to(x.device)
        filtered_noise = self.filter_bank.convolve(raw_noise)
        noise = noise * filtered_noise
        noise = noise.sum(dim=1, keepdim=True)

        harmonic = self.to_frames(x)
        harmonic = self.filter_bank.transposed_convolve(harmonic)

        x = harmonic + noise
        return x


class FilterBankGenerator(nn.Module):
    def __init__(self, filter_bank, in_size, out_size, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_size = out_size
        self.in_size = in_size
        self.channels = 256

        self._filter_bank = [filter_bank]
        self.main = UpsamplingStack(
            self.in_size,
            self.out_size,
            2,
            self._build_layer)
        self.to_frames = nn.Conv1d(
            self.channels, self.filter_bank.n_bands, 7, 1, 3)

    def _build_layer(self, i, curr_size, out_size, first, last):
        return LearnedUpSample(
            self.in_channels if first else self.channels,
            self.channels,
            8,
            2,
            lambda x: F.leaky_relu(x, 0.2))

    def to(self, device):
        self.filter_bank.to(device)
        return super().to(device)

    @property
    def filter_bank(self):
        return self._filter_bank[0]

    def forward(self, x):
        x = self.main(x)
        x = self.to_frames(x)
        x = self.filter_bank.transposed_convolve(x)
        return x
