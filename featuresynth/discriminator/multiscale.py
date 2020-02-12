from torch import nn
from ..audio.transform import fft_frequency_decompose
from torch.nn.init import xavier_normal_, calculate_gain
from torch.nn import functional as F
import torch


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
    def __init__(self):
        super().__init__()
        self.spec = {
            16384: {
                'scale_factors': [4, 4, 4, 4],
                'channels': [1, 32, 64, 128, 256]
            },
            8192: {
                'scale_factors': [4, 4, 4, 2],
                'channels': [1, 32, 64, 128, 256]
            },
            4096: {
                'scale_factors': [4, 4, 2, 2],
                'channels': [1, 32, 64, 128, 256]
            },
            2048: {
                'scale_factors': [4, 2, 2, 2],
                'channels': [1, 32, 64, 128, 256]
            },
            1024: {
                'scale_factors': [2, 2, 2, 2],
                'channels': [1, 32, 64, 128, 256]
            }
        }
        self.channel_discs = {}
        for key, value in self.spec.items():
            disc = ChannelDiscriminator(**value)
            self.add_module(f'channel_{key}', disc)
            self.channel_discs[key] = disc

        final_channels = sum(v['channels'][-1] for v in self.spec.values())
        self.judge = nn.Conv1d(final_channels, 1, 3, 1, 1)

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
        channels = []
        bands = fft_frequency_decompose(x, 1024)

        for size, layer in self.channel_discs.items():
            f, x = layer(bands[size])
            features.extend(f)
            channels.append(x)

        x = torch.cat(channels, dim=1)
        x = self.judge(x)
        return features, x
