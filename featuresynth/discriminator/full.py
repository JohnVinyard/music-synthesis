import zounds
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import calculate_gain, xavier_normal_
from featuresynth.util.modules import DilatedStack
import numpy as np
from torch.nn.utils import weight_norm


def weight_norm(x):
    return x


class FullDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            weight_norm(nn.Conv1d(1, 16, 15, 1, padding=7)),
            weight_norm(nn.Conv1d(16, 64, 41, 4, padding=20, groups=4)),
            weight_norm(nn.Conv1d(64, 256, 41, 4, padding=20, groups=16)),
            weight_norm(nn.Conv1d(256, 1024, 41, 4, padding=20, groups=64)),
            weight_norm(nn.Conv1d(1024, 1024, 41, 4, padding=20, groups=256)),
            weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2))
        )

        self.judge = weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def initialize_weights(self):
        for name, weight in self.named_parameters():
            if weight.data.dim() > 2:
                if 'judge' in name:
                    xavier_normal_(weight.data, 1)
                else:
                    xavier_normal_(
                        weight.data, calculate_gain('leaky_relu', 0.2))
        return self

    def forward(self, x):
        features = []
        for layer in self.main:
            x = F.leaky_relu(layer(x), 0.2)
            features.append(x)
        x = self.judge(x)
        return features, x


# class FilterBankDiscriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         sr = zounds.SR11025()
#         n_taps = 128
#         n_bands = 128
#         scale = zounds.MelScale(zounds.FrequencyBand(20, sr.nyquist), n_bands)
#         fb = zounds.learn.FilterBank(
#             samplerate=sr,
#             kernel_size=n_taps,
#             scale=scale,
#             scaling_factors=np.linspace(0.25, 0.5, len(scale)),
#             normalize_filters=False,
#             a_weighting=False)
#
#         self.fb = [fb]
#
#         self.main = nn.Sequential(
#             nn.Conv1d(n_bands, 128, 7, 4, 3, bias=False),
#             nn.Conv1d(128, 256, 7, 4, 3, bias=False),
#             nn.Conv1d(256, 512, 7, 4, 3, bias=False),
#             nn.Conv1d(512, 1024, 7, 4, 3, bias=False),
#         )
#
#         self.judge = nn.Conv1d(1024, 1, 3, 1, 1, bias=False)
#
#         self.ds = DilatedStack(
#             n_bands,
#             channels=128,
#             kernel_size=3,
#             dilations=[1, 3, 9],
#             activation=lambda x: F.leaky_relu(x, 0.2))
#         self.d_judge = nn.Conv1d(128, 1, 3, 1, 1, bias=False)
#
#     def to(self, device):
#         self.filter_bank.to(device)
#         return super().to(device)
#
#     @property
#     def filter_bank(self):
#         return self.fb[0]
#
#     def initialize_weights(self):
#         for name, weight in self.named_parameters():
#             if weight.data.dim() > 2:
#                 if 'judge' in name:
#                     xavier_normal_(weight.data, 1)
#                 else:
#                     xavier_normal_(
#                         weight.data, calculate_gain('leaky_relu', 0.2))
#         return self
#
#     def forward(self, x):
#
#         features = []
#         with_phase = x = \
#             self.filter_bank.forward(x, normalize=False)[...,
#             :x.shape[-1]].contiguous()
#         with_phase = F.relu(with_phase)
#         for layer in self.main:
#             x = F.leaky_relu(layer(x), 0.2)
#             features.append(x)
#         with_phase_j = self.judge(x)
#         x = self.filter_bank.temporal_pooling(with_phase, 512, 256)[...,
#             :64].contiguous()
#         f, x = self.ds.forward(x, return_features=True)
#         features.extend(f)
#         x = self.d_judge(x)
#         x = torch.cat([with_phase_j, x], dim=1)
#         return features, x
