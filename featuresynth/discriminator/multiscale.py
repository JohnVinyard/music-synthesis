from torch import nn
from ..audio.transform import fft_frequency_decompose
from ..util.modules import DownsamplingStack
from torch.nn.init import xavier_normal_, calculate_gain
from torch.nn import functional as F
from functools import reduce
import torch
import numpy as np


class STFTDiscriminator(nn.Module):
    def __init__(self, start_size, target_size):
        super().__init__()
        self.target_size = target_size
        self.start_size = start_size
        self.channels = [256, 512, 1024, 2048]
        self.main = DownsamplingStack(
            start_size=self.start_size,
            target_size=self.target_size,
            scale_factor=2,
            layer_func=self._build_layer,
            activation=lambda x: F.leaky_relu(x, 0.2))
        self.judge = nn.Conv1d(self.channels[-1], 1, 7, 1, 3)

    # def initialize_weights(self):
    #     for name, weight in self.named_parameters():
    #         if weight.data.dim() > 2:
    #             if 'judge' in name:
    #                 xavier_normal_(weight.data, 1)
    #             else:
    #                 xavier_normal_(
    #                     weight.data, calculate_gain('leaky_relu', 0.2))
    #     return self

    def _build_layer(self, i, curr_size, out_size, first, last):
        return nn.Conv1d(
            self.channels[i],
            self.channels[i + 1],
            7,
            2,
            3)

    def forward(self, x):
        batch, channels, time = x.shape
        x = F.pad(x, (0, 256))
        x = torch.stft(
            x.view(batch, -1),
            n_fft=512,
            hop_length=256,
            win_length=512,
            normalized=True,
            center=False)

        x = torch.abs(x[:, 1:, :, 0])
        features, x = self.main(x, return_features=True)
        x = self.judge(x)
        return [features], [x]


class ChannelDiscriminator(nn.Module):
    def __init__(self, scale_factors, channels):
        super().__init__()
        self.channels = channels
        self.scale_factors = scale_factors
        layers = []
        for i in range(len(scale_factors)):
            layers.append(nn.Conv1d(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=9,
                stride=scale_factors[i],
                padding=4))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        features = []
        for layer in self.main:
            x = F.leaky_relu(layer(x), 0.2)
            features.append(x)
        return features, x


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.input_size = input_size
        band_sizes = \
            [int(2 ** (np.log2(self.input_size) - i)) for i in range(5)]
        spec_template = {
            0: {
                'scale_factors': [4, 4, 4, 4],
                'channels': [1, 32, 64, 128, 256]
            },
            1: {
                'scale_factors': [4, 4, 4, 2],
                'channels': [1, 32, 64, 128, 256]
            },
            2: {
                'scale_factors': [4, 4, 2, 2],
                'channels': [1, 32, 64, 128, 256]
            },
            3: {
                'scale_factors': [4, 2, 2, 2],
                'channels': [1, 32, 64, 128, 256]
            },
            4: {
                'scale_factors': [2, 2, 2, 2],
                'channels': [1, 32, 64, 128, 256]
            }
        }

        # produce keys in descending order of band size, e.g.:
        # [8192, 4096, 2048, 1024, 512]
        self.spec = {bs: v for bs, v in zip(band_sizes, spec_template.values())}

        self.smallest_band = min(self.spec.keys())

        self.channel_discs = {}
        for key, value in self.spec.items():
            disc = ChannelDiscriminator(**value)
            self.add_module(f'channel_{key}', disc)
            self.channel_discs[key] = disc

        final_channels = sum(v['channels'][-1] for v in self.spec.values())
        self.judge = nn.Conv1d(final_channels, 1, 3, 1, 1)

    # def initialize_weights(self):
    #     for name, weight in self.named_parameters():
    #         if weight.data.dim() > 2:
    #             if 'judge' in name:
    #                 xavier_normal_(weight.data, 1)
    #             else:
    #                 xavier_normal_(
    #                     weight.data, calculate_gain('leaky_relu', 0.2))
    #     return self

    def forward(self, x):
        features = []
        channels = []
        bands = fft_frequency_decompose(x, self.smallest_band)

        for size, layer in self.channel_discs.items():
            f, x = layer(bands[size])
            features.append(f)
            channels.append(x)

        x = torch.cat(channels, dim=1)
        x = self.judge(x)
        return features, [x]


class MultiScaleMultiResDiscriminator(nn.Module):
    def __init__(self, input_size, flatten_multiscale_features=False):
        super().__init__()
        self.input_size = input_size
        self.flatten_multiscale_features = flatten_multiscale_features
        self.multiscale = MultiScaleDiscriminator(input_size)

        hop_size = 256
        low_res_input_size = input_size // hop_size
        self.low_res = STFTDiscriminator(
            low_res_input_size, low_res_input_size // 8)

    # def initialize_weights(self):
    #     for name, weight in self.named_parameters():
    #         if weight.data.dim() > 2:
    #             if 'judge' in name:
    #                 xavier_normal_(weight.data, 1)
    #             else:
    #                 xavier_normal_(
    #                     weight.data, calculate_gain('leaky_relu', 0.2))
    #     return self

    def forward(self, x):
        features = []
        judgements = []

        f, j = self.multiscale(x)
        if self.flatten_multiscale_features:
            # treat features from each band as a single group so they don't
            # dominate the feature-matching loss function
            f = reduce(lambda a, b: a + b, f, [])
            features.append(f)
        else:
            features.extend(f)
        judgements.extend(j)

        f, j = self.low_res(x)
        features.extend(f)
        judgements.extend(j)

        return features, judgements
