from torch import nn
from torch.nn.init import xavier_normal_, calculate_gain
from torch.nn import functional as F
import torch
from ..util.modules import DownsamplingStack
import numpy as np


class LowResSpectrogramDiscriminator(nn.Module):
    def __init__(
            self,
            freq_bins,
            time_steps,
            n_judgements,
            kernel_size,
            max_channels):
        super().__init__()
        self.max_channels = max_channels
        self.kernel_size = kernel_size
        self.n_judgements = n_judgements
        self.time_steps = time_steps
        self.freq_bins = freq_bins
        self.stack = DownsamplingStack(
            start_size=time_steps,
            target_size=n_judgements,
            scale_factor=2,
            layer_func=self._build_layer,
            activation=lambda x: F.leaky_relu(x, 0.2))
        self.judge = nn.Conv1d(self.stack.out_channels, 1, 3, 1, 1)

    def _build_layer(self, i, curr_size, out_size, first, last):
        log_channels = np.log2(self.freq_bins)
        in_channels = min(self.max_channels, 2 ** (i + log_channels))
        out_channels = min(self.max_channels, 2 ** (i + log_channels + 1))
        return nn.Conv1d(
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            kernel_size=self.kernel_size,
            stride=2,
            padding=self.kernel_size // 2)

    def forward(self, x):
        batch, channels, time = x.shape
        channel_window = channels // self.freq_bins
        time_window = time // self.time_steps
        low_res = F.avg_pool2d(
            F.relu(x)[:, None, :, :],
            (channel_window, time_window))
        low_res = low_res.view(-1, self.freq_bins, self.time_steps)
        features = []
        for layer in self.stack:
            low_res = F.leaky_relu(layer(low_res), 0.2)
            features.append(low_res)
        j = self.judge(low_res)
        return features, j


class FilterBankDiscriminator(nn.Module):
    def __init__(self, filter_bank):
        super().__init__()
        self._filter_bank = [filter_bank]

        in_channels = self.filter_bank.filter_bank.shape[0]
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, 256, 7, 2, 3),
            nn.Conv1d(256, 256, 7, 2, 3),
            nn.Conv1d(256, 512, 7, 2, 3),
            nn.Conv1d(512, 512, 7, 2, 3),
            nn.Conv1d(512, 1024, 7, 2, 3),
            nn.Conv1d(1024, 1024, 7, 2, 3),
            nn.Conv1d(1024, 1024, 7, 2, 3),
            nn.Conv1d(1024, 1024, 7, 2, 3))
        self.judge = nn.Conv1d(1024, 1, 3, 1, 1)

        self.medium_res = LowResSpectrogramDiscriminator(
            freq_bins=128,
            time_steps=128,
            n_judgements=16,
            kernel_size=7,
            max_channels=1024)

        self.low_res = LowResSpectrogramDiscriminator(
            freq_bins=32,
            time_steps=32,
            n_judgements=4,
            kernel_size=7,
            max_channels=512)

    @property
    def filter_bank(self):
        return self._filter_bank[0]

    def to(self, device):
        self.filter_bank.to(device)
        return super().to(device)

    def initialize_weights(self):
        for name, weight in self.named_parameters():
            if weight.data.dim() > 2:
                if 'judge' in name:
                    xavier_normal_(weight.data, 1)
                else:
                    xavier_normal_(
                        weight.data, calculate_gain('leaky_relu', 0.2))
        return self

    def full_resolution(self, x):
        features = []
        for layer in self.main:
            x = F.leaky_relu(layer(x), 0.2)
            features.append(x)
        x = self.judge(x)
        return features, x

    def forward(self, x):
        batch = x.shape[0]

        x = self.filter_bank.convolve(x)

        features = []
        judgements = []

        # f, j = self.full_resolution(x)
        # features.extend(f)
        # judgements.append(j)

        f, j = self.medium_res(x)
        features.extend(f)
        judgements.append(j)

        # f, j = self.low_res(x)
        # features.extend(f)
        # judgements.append(j)

        x = torch.cat([j.view(batch, -1) for j in judgements], dim=-1)
        return features, x
