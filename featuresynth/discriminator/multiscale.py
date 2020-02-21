from torch import nn
from ..audio.transform import fft_frequency_decompose
from ..util.modules import DownsamplingStack
from torch.nn.init import xavier_normal_, calculate_gain
from torch.nn import functional as F
import torch


class STFTDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels = [256, 512, 1024, 2048]
        self.main = DownsamplingStack(
            64,
            8,
            2,
            layer_func=self._build_layer,
            activation=lambda x: F.leaky_relu(x, 0.2))
        self.judge = nn.Conv1d(self.channels[-1], 1, 7, 1, 3)

    def _build_layer(self, i, curr_size, out_size, first, last):
        return nn.Conv1d(
            self.channels[i],
            self.channels[i + 1],
            7,
            1,
            3)

    def forward(self, x):
        batch, channels, time = x.shape
        x = torch.stft(
            x.view(batch, time),
            frame_length=512,
            hop=256,
            fft_size=512,
            normalized=True,
            pad_end=256)
        x = torch.abs(x[:, :, 1:, 0])
        x = x.permute(0, 2, 1).contiguous()
        features, x = self.main(x, return_features=True)
        x = self.judge(x)
        return features, [x]


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
            features.append(f)
            channels.append(x)

        x = torch.cat(channels, dim=1)
        x = self.judge(x)
        return features, [x]


class MultiScaleMultiResDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.multiscale = MultiScaleDiscriminator()
        self.low_res = STFTDiscriminator()

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
        judgements = []

        f, j = self.multiscale(x)
        features.extend(f)
        judgements.extend(j)

        f, j = self.low_res(x)
        features.append(f)
        judgements.extend(j)

        return features, judgements
