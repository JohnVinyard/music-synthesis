from torch import nn
from torch.nn.init import xavier_normal_, calculate_gain
from torch.nn import functional as F
from ..util.modules import LowResSpectrogramDiscriminator
import torch


class MDCTDiscriminator(nn.Module):
    def __init__(self, in_channels, feature_size, conditioning_channels=0):
        super().__init__()
        self.conditioning_channels = conditioning_channels
        self.feature_size = feature_size
        self.in_channels = in_channels
        self.main = nn.Sequential(
            nn.Conv1d(in_channels + self.conditioning_channels, 512, 7, 1, 3),
            nn.Conv1d(512, 1024, 7, 2, 3, groups=4),
            nn.Conv1d(1024, 1024, 7, 2, 3, groups=16),
            nn.Conv1d(1024, 1024, 7, 2, 3, groups=32),
        )
        self.judge = nn.Conv1d(1024, 1, 3, 1, 1)

        self.med_res = LowResSpectrogramDiscriminator(
            freq_bins=64,
            time_steps=feature_size // 2,
            n_judgements=4,
            kernel_size=7,
            max_channels=1024,
            conditioning_channels=self.conditioning_channels)

        self.low_res = LowResSpectrogramDiscriminator(
            freq_bins=32,
            time_steps=feature_size // 4,
            n_judgements=2,
            kernel_size=3,
            max_channels=1024,
            conditioning_channels=self.conditioning_channels)



    def full_resolution(self, x, feat):
        features = []
        if self.conditioning_channels > 0:
            x = torch.cat([x, feat], axis=1)
        for layer in self.main:
            x = F.leaky_relu(layer(x), 0.2)
            features.append(x)
        j = self.judge(x)
        return features, j

    def forward(self, x, feat):
        features = []
        judgements = []

        f, j = self.med_res(x, feat)
        features.append(f)
        judgements.append(j)

        f, j = self.low_res(x, feat)
        features.append(f)
        judgements.append(j)

        f, j = self.full_resolution(x, feat)
        features.append(f)
        judgements.append(j)

        return features, judgements


