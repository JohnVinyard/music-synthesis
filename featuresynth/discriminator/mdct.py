from torch import nn
from torch.nn.init import xavier_normal_, calculate_gain
from torch.nn import functional as F
from ..util.modules import LowResSpectrogramDiscriminator
import torch


class MDCTDiscriminator(nn.Module):
    def __init__(self, in_channels, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.in_channels = in_channels
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, 512, 7, 1, 3),
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
            max_channels=1024)

        self.low_res = LowResSpectrogramDiscriminator(
            freq_bins=32,
            time_steps=feature_size // 4,
            n_judgements=2,
            kernel_size=3,
            max_channels=1024)

    # def initialize_weights(self):
    #     for name, weight in self.named_parameters():
    #         if weight.data.dim() > 2:
    #             if 'judge' in name:
    #                 xavier_normal_(weight.data, calculate_gain('tanh'))
    #             else:
    #                 xavier_normal_(
    #                     weight.data, calculate_gain('leaky_relu', 0.2))
    #     return self

    def full_resolution(self, x):
        features = []
        for layer in self.main:
            x = F.leaky_relu(layer(x), 0.2)
            features.append(x)
        j = self.judge(x)
        return features, j

    def forward(self, x):
        features = []
        judgements = []

        f, j = self.med_res(x)
        features.append(f)
        judgements.append(j)

        f, j = self.low_res(x)
        features.append(f)
        judgements.append(j)

        f, j = self.full_resolution(x)
        features.append(f)
        judgements.append(j)

        return features, judgements


class TwoDimMDCTDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), (2, 2), (1, 1)),
            nn.Conv2d(16, 32, (3, 3), (2, 2), (1, 1)),
            nn.Conv2d(32, 64, (3, 3), (2, 2), (1, 1)),
            nn.Conv2d(64, 128, (3, 3), (2, 2), (1, 1)),
            nn.Conv2d(128, 256, (3, 3), (2, 1), (1, 1)),
            nn.Conv2d(256, 512, (3, 3), (2, 1), (1, 1)),
        )

        self.judge = nn.Conv2d(512, 1, (4, 4), (1, 1), (0, 0))

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
        x = x[:, None, :, :]
        features = []
        for layer in self.main:
            x = F.leaky_relu(layer(x), 0.2)
            features.append(x)
        x = self.judge(x)
        return features, x
