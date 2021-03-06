from torch import nn
from torch.nn.init import xavier_normal_, calculate_gain
from torch.nn import functional as F
import torch
from ..util.modules import LowResSpectrogramDiscriminator
from torch.nn.utils import weight_norm


# class LowResFilterBankDiscriminator(nn.Module):
#     def __init__(self, filter_bank):
#         super().__init__()
#         self._filter_bank = [filter_bank]
#
#         self.low_res = LowResSpectrogramDiscriminator(
#             freq_bins=64,
#             time_steps=64,
#             n_judgements=8,
#             kernel_size=9,
#             max_channels=1024)
#
#     @property
#     def filter_bank(self):
#         return self._filter_bank[0]
#
#     def to(self, device):
#         self.filter_bank.to(device)
#         return super().to(device)
#
#     def forward(self, x):
#         batch = x.shape[0]
#
#         x = self.filter_bank.convolve(x)
#
#         features = []
#         judgements = []
#
#         f, j = self.low_res(x)
#         features.extend(f)
#         judgements.append(j)
#
#         x = torch.cat([j.view(batch, -1) for j in judgements], dim=-1)
#         return features, x


class LargeReceptiveFieldFilterBankDiscriminator(nn.Module):
    def __init__(self, filter_bank, add_weight_norm=True):
        super().__init__()
        self.add_weight_norm = add_weight_norm
        self.filter_bank = filter_bank

        in_channels = self.filter_bank.filter_bank.shape[0]
        self.main = nn.Sequential(
            self._conv_layer(in_channels, 256, 15, 1, 7),
            self._conv_layer(256, 256, 41, 4, 20, groups=4),
            self._conv_layer(256, 512, 41, 4, 20, groups=16),
            self._conv_layer(512, 1024, 41, 4, 20, groups=64),
            self._conv_layer(1024, 1024, 41, 4, 20, groups=256),
            self._conv_layer(1024, 1024, 5, 1, 2)
        )
        self.judge = self._conv_layer(1024, 1, 3, 1, 1)

        # self.medium_res = LowResSpectrogramDiscriminator(
        #     freq_bins=128,
        #     time_steps=128,
        #     n_judgements=16,
        #     kernel_size=15,
        #     max_channels=1024)
        #
        # self.low_res = LowResSpectrogramDiscriminator(
        #     freq_bins=32,
        #     time_steps=32,
        #     n_judgements=4,
        #     kernel_size=9,
        #     max_channels=512)

    def _conv_layer(self, *args, **kwargs):
        conv = nn.Conv1d(*args, **kwargs)
        if self.add_weight_norm:
            conv = weight_norm(conv)
        return conv

    def full_resolution(self, x):
        features = []
        for layer in self.main:
            x = F.leaky_relu(layer(x), 0.2)
            features.append(x)
        x = self.judge(x)
        return features, x

    def forward(self, x, feat):
        batch = x.shape[0]

        x = self.filter_bank.convolve(x)

        features = []
        judgements = []

        f, j = self.full_resolution(x)
        features.extend(f)
        judgements.append(j)

        # f, j = self.medium_res(x)
        # features.extend(f)
        # judgements.append(j)
        #
        # f, j = self.low_res(x)
        # features.extend(f)
        # judgements.append(j)

        x = torch.cat([j.view(batch, -1) for j in judgements], dim=-1)
        return features, x


class FilterBankDiscriminator(nn.Module):
    def __init__(
            self,
            filter_bank,
            input_size,
            conditioning_channels=0,
            log_scaling=False):

        super().__init__()
        self.log_scaling = log_scaling
        self.conditioning_channels = conditioning_channels
        self.input_size = input_size
        self._filter_bank = [filter_bank]

        # in_channels = self.filter_bank.filter_bank.shape[0]
        in_channels = self.filter_bank.n_bands
        self.main = nn.Sequential(
            nn.Conv1d(in_channels + conditioning_channels, 256, 7, 2, 3),
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
            max_channels=1024,
            conditioning_channels=conditioning_channels,
            log_scaling=log_scaling)

        self.low_res = LowResSpectrogramDiscriminator(
            freq_bins=32,
            time_steps=32,
            n_judgements=4,
            kernel_size=7,
            max_channels=512,
            conditioning_channels=conditioning_channels,
            log_scaling=log_scaling)

    @property
    def filter_bank(self):
        return self._filter_bank[0]

    def to(self, device):
        self.filter_bank.to(device)
        return super().to(device)

    def full_resolution(self, x, feat):

        if self.conditioning_channels > 0:
            # TODO: Consider the alternative (and less memory-intensive) option
            # adding together unconditioned and computed features, as seen here:
            # https://arxiv.org/pdf/1909.11646.pdf#page=5
            # https://github.com/yanggeng1995/GAN-TTS/blob/master/models/discriminator.py#L67
            feat = F.upsample(feat, size=x.shape[-1])
            x = torch.cat([x, feat], dim=1)

        features = []
        for layer in self.main:
            x = F.leaky_relu(layer(x), 0.2)
            features.append(x)
        x = self.judge(x)
        return features, x

    def forward(self, x, feat):
        x = self.filter_bank.convolve(x)

        features = []
        judgements = []

        f, j = self.full_resolution(x, feat)
        features.append(f)
        judgements.append(j)

        f, j = self.medium_res(x, feat)
        features.append(f)
        judgements.append(j)

        f, j = self.low_res(x, feat)
        features.append(f)
        judgements.append(j)

        return features, judgements
