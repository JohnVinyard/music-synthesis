from torch import nn
from ..util.modules import LearnedUpSample, DilatedStack, UpSample
from ..audio.transform import fft_frequency_recompose
from torch.nn import functional as F
import numpy as np


class ChannelGenerator(nn.Module):
    def __init__(self, scale_factors, channels, transposed_conv=False):
        super().__init__()
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
                    scale_factor=scale_factors[i],
                    activation=lambda x: F.leaky_relu(x, 0.2)))
            else:
                layers.append(UpSample(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=scale_factors[i] * 2 + 1,
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


class MultiScaleGenerator(nn.Module):
    def __init__(
            self,
            feature_channels,
            input_size,
            output_size,
            transposed_conv=False,
            recompose=True):

        super().__init__()

        self.recompose = recompose
        self.input_size = input_size
        self.output_size = output_size
        self.feature_channels = feature_channels
        self.embedding = nn.Conv1d(feature_channels, 512, 7, 1, 3)
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
