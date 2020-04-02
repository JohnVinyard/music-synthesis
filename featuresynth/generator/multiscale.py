from torch import nn
from ..util.modules import LearnedUpSample, DilatedStack, UpSample
from ..audio.transform import fft_frequency_recompose
from torch.nn import functional as F
import numpy as np
import torch
import zounds


class ChannelGenerator(nn.Module):
    def __init__(
            self,
            scale_factors,
            channels,
            transposed_conv=False,
            kernel_size=40):

        super().__init__()
        self.kernel_size = kernel_size
        self.transposed_conv = transposed_conv
        self.channels = channels
        self.scale_factors = scale_factors
        layers = []
        for i in range(len(scale_factors)):
            if self.transposed_conv:
                layers.append(LearnedUpSample(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=scale_factors[i] * 2,
                    # kernel_size=40,
                    # kernel_size=self.kernel_size,
                    scale_factor=scale_factors[i],
                    activation=lambda x: F.leaky_relu(x, 0.2)))
            else:
                layers.append(UpSample(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=scale_factors[i] * 2 + 1,
                    # kernel_size=41,
                    # kernel_size=self.kernel_size + 1,
                    scale_factor=scale_factors[i],
                    activation=lambda x: F.leaky_relu(x, 0.2)))
            layers.append(DilatedStack(
                channels[i + 1],
                channels[i + 1],
                3,
                [1, 3, 9],
                activation=lambda x: F.leaky_relu(x, 0.2),
                residual=True))
        self.main = nn.Sequential(*layers)
        self.to_samples = nn.Conv1d(channels[-1], 1, 7, 1, 3)

    def forward(self, x):
        x = self.main(x)
        x = self.to_samples(x)
        return x


class FilterBankChannelGenerator(nn.Module):
    def __init__(self, scale_factors, channels, filter_bank):
        super().__init__()

        self.filter_bank = filter_bank
        self.channels = channels
        self.scale_factors = scale_factors

        layers = []
        for i in range(len(scale_factors)):
            if i == 0:
                # embedding layer
                layers.append(nn.Sequential(
                    nn.Conv1d(channels[i], channels[i + 1], 7, 1, 3),
                    nn.LeakyReLU(0.2)
                ))

            else:
                # upscaling layer
                layers.append(LearnedUpSample(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=scale_factors[i] * 2,
                    scale_factor=scale_factors[i],
                    activation=lambda x: F.leaky_relu(x, 0.2)))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        x = self.main(x)
        # TODO: Consider multi-head architecture that also produces filtered
        # noise
        x = F.pad(x, (0, 1))
        x = self.filter_bank.transposed_convolve(x)
        return x


class FilterBankMultiScaleGenerator(nn.Module):
    def __init__(
            self,
            samplerate,
            feature_channels,
            input_size,
            output_size,
            recompose=True):

        super().__init__()
        self.samplerate = samplerate
        self.recompose = recompose
        self.output_size = output_size
        self.input_size = input_size
        self.feature_channels = feature_channels
        band_sizes = [int(2 ** (np.log2(output_size) - i)) for i in range(5)]
        self.upsample_ratio = output_size // input_size

        spec_template = {
            0: {
                'scale_factors': [1, 4, 4, 4, 4],
                'channels': [128] * 6,
                'filter_bank': self._filter_bank(samplerate, 128)
            },
            1: {
                'scale_factors': [1, 4, 4, 4, 2],
                'channels': [128] * 6,
                'filter_bank': self._filter_bank(samplerate * 2, 128)
            },
            2: {
                'scale_factors': [1, 4, 4, 2, 2],
                'channels': [128] * 6,
                'filter_bank': self._filter_bank(samplerate * 4, 128)
            },
            3: {
                'scale_factors': [1, 4, 2, 2, 2],
                'channels': [128] * 6,
                'filter_bank': self._filter_bank(samplerate * 8, 128)
            },
            4: {
                'scale_factors': [1, 2, 2, 2, 2],
                'channels': [128] * 6,
                'filter_bank': self._filter_bank(
                    samplerate * 16, 128, zero_start=True)
            }
        }
        # produce keys in descending order of band size, e.g.:
        # [8192, 4096, 2048, 1024, 512]
        self.spec = {bs: v for bs, v in zip(band_sizes, spec_template.values())}

        self.channel_generators = {}
        for key, value in self.spec.items():
            generator = FilterBankChannelGenerator(**value)
            self.add_module(f'channel_{key}', generator)
            self.channel_generators[key] = generator

    def _scale(self, samplerate, bands, zero_start=False):
        start = 0 if zero_start else samplerate.nyquist / 2
        end = samplerate.nyquist
        return zounds.LinearScale(zounds.FrequencyBand(start, end), bands)

    def _filter_bank(self, samplerate, bands, zero_start=False):
        fb = zounds.learn.FilterBank(
            samplerate=samplerate,
            kernel_size=128,
            scale=self._scale(samplerate, bands, zero_start=zero_start),
            scaling_factors=0.05,
            normalize_filters=True,
            a_weighting=False)
        return fb

    def forward(self, x):
        input_size = x.shape[-1]

        results = {}
        for size, layer in self.channel_generators.items():
            results[size] = layer(x)

        if self.recompose:
            final = fft_frequency_recompose(
                results, input_size * self.upsample_ratio)
            return final
        else:
            return results

class MultiScaleGenerator(nn.Module):
    def __init__(
            self,
            feature_channels,
            input_size,
            output_size,
            transposed_conv=False,
            recompose=True,
            kernel_size=40):

        super().__init__()

        self.kernel_size = kernel_size
        self.recompose = recompose
        self.input_size = input_size
        self.output_size = output_size
        self.feature_channels = feature_channels
        self.embedding = nn.Conv1d(feature_channels, 512, 7, 1, padding=0)
        self.upsample_ratio = output_size // input_size

        band_sizes = [int(2 ** (np.log2(output_size) - i)) for i in range(5)]

        spec_template = {
            0: {
                'scale_factors': [4, 4, 4, 4],
                'channels': [512, 256, 128, 64, 32]
            },
            1: {
                'scale_factors': [4, 4, 4, 2],
                'channels': [512, 256, 128, 64, 32]
            },
            2: {
                'scale_factors': [4, 4, 2, 2],
                'channels': [512, 256, 128, 64, 32]
            },
            3: {
                'scale_factors': [4, 2, 2, 2],
                'channels': [512, 256, 128, 64, 32]
            },
            4: {
                'scale_factors': [2, 2, 2, 2],
                'channels': [512, 256, 128, 64, 32]
            }
        }
        # produce keys in descending order of band size, e.g.:
        # [8192, 4096, 2048, 1024, 512]
        self.spec = {bs: v for bs, v in zip(band_sizes, spec_template.values())}

        self.channel_generators = {}
        for key, value in self.spec.items():
            generator = ChannelGenerator(
                **value, transposed_conv=transposed_conv)
            self.add_module(f'channel_{key}', generator)
            self.channel_generators[key] = generator


    def forward(self, x):
        input_size = x.shape[-1]

        x = torch.nn.ReflectionPad1d(3).forward(x)
        x = F.leaky_relu(self.embedding(x), 0.2)

        results = {}
        for size, layer in self.channel_generators.items():
            results[size] = layer(x)

        if self.recompose:
            final = fft_frequency_recompose(
                results, input_size * self.upsample_ratio)
            return final
        else:
            return results


if __name__ == '__main__':
    import torch
    feature_channels = 256
    inp = torch.ones((2, feature_channels, 64))
    network = MultiScaleGenerator(feature_channels)
    result = network.forward(inp)
    print(result.shape)
