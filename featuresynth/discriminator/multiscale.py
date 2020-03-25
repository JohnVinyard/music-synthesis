from torch import nn
from ..audio.transform import fft_frequency_decompose
from ..util.modules import STFTDiscriminator
from torch.nn import functional as F
from functools import reduce
import torch
import numpy as np



class ChannelDiscriminator(nn.Module):
    def __init__(
            self,
            scale_factors,
            channels,
            return_judgements=False,
            conditioning_channels=0):

        super().__init__()
        self.conditioning_channels = conditioning_channels
        self.return_judgements = return_judgements
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

        if self.return_judgements:
            self.judge = nn.Conv1d(channels[-1], 1, 3, 1, 1)

    def forward(self, x, feat):
        features = []



        for layer in self.main:
            x = F.leaky_relu(layer(x), 0.2)
            features.append(x)
        if self.return_judgements:
            j = self.judge(x)
            return features, x, j
        else:
            return features, x


class MultiScaleDiscriminator(nn.Module):
    def __init__(
            self,
            input_size,
            decompose=True,
            channel_judgements=False,
            conditioning_channels=0):

        super().__init__()

        self.conditioning_channels = conditioning_channels
        self.channel_judgements = channel_judgements
        self.decompose = decompose
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
            disc = ChannelDiscriminator(
                **value,
                return_judgements=self.channel_judgements,
                conditioning_channels=self.conditioning_channels)
            self.add_module(f'channel_{key}', disc)
            self.channel_discs[key] = disc

        final_channels = sum(v['channels'][-1] for v in self.spec.values())
        channels = 512
        self.final = nn.Sequential(
            nn.Conv1d(final_channels + self.conditioning_channels, channels, 3, 1, 1),
            nn.Conv1d(channels, channels, 3, 1, 1),
            nn.Conv1d(channels, channels, 3, 1, 1),
        )
        self.judge = nn.Conv1d(channels, 1, 3, 1, 1)


    def forward(self, x, feat):
        features = []
        channels = []
        judgements = []

        if self.decompose:
            bands = fft_frequency_decompose(x, self.smallest_band)
        else:
            bands = x

        for size, layer in self.channel_discs.items():
            if self.channel_judgements:
                f, x, j = layer(bands[size], feat)
                features.append(f)
                channels.append(x)
                judgements.append(j)
            else:
                f, x = layer(bands[size])
                features.append(f)
                channels.append(x)

        x = torch.cat(channels, dim=1)

        if self.conditioning_channels > 0:
            # TODO: Consider the alternative (and less memory-intensive) option
            # adding together unconditioned and computed features, as seen here:
            # https://arxiv.org/pdf/1909.11646.pdf#page=5
            # https://github.com/yanggeng1995/GAN-TTS/blob/master/models/discriminator.py#L67
            feat = F.upsample(feat, size=x.shape[-1])
            x = torch.cat([x, feat], dim=1)

        final_features = []
        for layer in self.final:
            x = F.leaky_relu(layer(x), 0.2)
            final_features.append(x)

        features.append(final_features)
        x = self.judge(x)
        judgements.append(x)
        return features, judgements


class MultiScaleMultiResDiscriminator(nn.Module):
    def __init__(
            self,
            input_size,
            flatten_multiscale_features=False,
            decompose=True,
            channel_judgements=False,
            conditioning_channels=0):

        super().__init__()
        self.conditioning_channels = conditioning_channels
        self.input_size = input_size
        self.flatten_multiscale_features = flatten_multiscale_features
        self.multiscale = MultiScaleDiscriminator(
            input_size, decompose, channel_judgements, conditioning_channels)

        hop_size = 256
        low_res_input_size = input_size // hop_size
        self.low_res = STFTDiscriminator(
            low_res_input_size, low_res_input_size // 8)


    def forward(self, x, feat):
        features = []
        judgements = []

        f, j = self.multiscale(x, feat)
        if self.flatten_multiscale_features:
            # treat features from each band as a single group so they don't
            # dominate the feature-matching loss function
            f = reduce(lambda a, b: a + b, f, [])
            features.append(f)
        else:
            features.extend(f)
        judgements.extend(j)

        # f, j = self.low_res(x)
        # features.extend(f)
        # judgements.extend(j)

        return features, judgements
