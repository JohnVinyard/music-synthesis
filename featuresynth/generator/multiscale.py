from torch import nn
from ..util.modules import LearnedUpSample, DilatedStack
from ..audio.transform import fft_frequency_recompose
from torch.nn.init import xavier_normal_, calculate_gain
from torch.nn import functional as F


class ChannelGenerator(nn.Module):
    def __init__(self, scale_factors, channels):
        super().__init__()
        self.channels = channels
        self.scale_factors = scale_factors
        layers = []
        for i in range(len(scale_factors)):
            layers.append(LearnedUpSample(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=scale_factors[i] * 2,
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
    def __init__(self, feature_channels):
        super().__init__()

        self.feature_channels = feature_channels
        self.embedding = nn.Conv1d(feature_channels, 512, 7, 1, 3)

        self.spec = {
            16384: {
                'scale_factors': [4, 4, 4, 4],
                'channels': [512, 256, 128, 64, 32]
            },
            8192: {
                'scale_factors': [4, 4, 4, 2],
                'channels': [512, 256, 128, 64, 32]
            },
            4096: {
                'scale_factors': [4, 4, 2, 2],
                'channels': [512, 256, 128, 64, 32]
            },
            2048: {
                'scale_factors': [4, 2, 2, 2],
                'channels': [512, 256, 128, 64, 32]
            },
            1024: {
                'scale_factors': [2, 2, 2, 2],
                'channels': [512, 256, 128, 64, 32]
            }
        }
        self.channel_generators = {}
        for key, value in self.spec.items():
            generator = ChannelGenerator(**value)
            self.add_module(f'channel_{key}', generator)
            self.channel_generators[key] = generator

    def initialize_weights(self):
        for name, weight in self.named_parameters():
            if weight.data.dim() > 2:
                if 'samples' in name:
                    xavier_normal_(weight.data, 1)
                else:
                    xavier_normal_(
                        weight.data, calculate_gain('leaky_relu', 0.2))
        return self

    def forward(self, x):
        x = F.leaky_relu(self.embedding(x), 0.2)

        results = {}
        for size, layer in self.channel_generators.items():
            results[size] = layer(x)

        final = fft_frequency_recompose(results, 16384)
        return final


if __name__ == '__main__':
    import torch
    feature_channels = 256
    inp = torch.ones((2, feature_channels, 64))
    network = MultiScaleGenerator(feature_channels)
    result = network.forward(inp)
    print(result.shape)
