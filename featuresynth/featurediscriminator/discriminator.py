import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, calculate_gain
import numpy as np
from ..util.modules import \
    DownsamplingStack, flatten_channels, unflatten_channels, DilatedStack
import math
import zounds


def shift_and_scale(x):
    orig_shape = x.shape
    x = x.view(x.shape[0], -1)

    # shift
    mn, _ = x.min(dim=1, keepdim=True)
    x = x - mn

    # scale
    mx, _ = x.max(dim=1, keepdim=True)
    x = x / (mx + 1e-12)
    x = x.view(*orig_shape)
    return x


class CausalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            bias=False)

    def forward(self, x):
        batch_size = x.shape[0]
        x = inp = x.view(batch_size, self.in_channels, -1)
        x = torch.cat([
            torch.zeros(batch_size, self.in_channels, self.padding).to(
                x.device),
            x
        ], dim=-1)
        x = self.conv(x)
        if x.shape[1] == inp.shape[1]:
            x = inp + x
        x = F.leaky_relu(x, 0.2)
        return x


def make_stack(start_size, target_size, layer_func):
    n_layers = int(np.log2(start_size) - np.log2(target_size))
    layers = []
    for i in range(n_layers):
        layers.append(layer_func(i))
    return nn.Sequential(*layers)


class ARDiscriminator(nn.Module):
    def __init__(self, frames, feature_channels, channels, n_judgements, ae):
        super().__init__()
        self.n_judgements = n_judgements
        self.channels = channels
        self.feature_channels = feature_channels
        self.frames = frames

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 128, 7, 2, 3, bias=False),
            nn.Conv1d(128, 128, 7, 2, 3, bias=False),
            nn.Conv1d(128, 128, 7, 2, 3, bias=False),
            nn.Conv1d(128, 128, 7, 2, 3, bias=False),
            nn.Conv1d(128, 128, 3, 2, 1, bias=False),
            nn.Conv1d(128, 128, 3, 2, 1, bias=False),
            nn.Conv1d(128, 128, 3, 2, 1, bias=False),
            nn.Conv1d(128, 128, 3, 2, 1, bias=False),
            nn.Conv1d(128, channels, 1, 1, 0, bias=False),
        )

        self.main = nn.Sequential(
            nn.Conv1d(channels, channels, 2, dilation=1, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=2, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=4, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=8, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=16, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=32, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=64, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=128, bias=False),
        )

        self.gate = nn.Sequential(
            nn.Conv1d(channels, channels, 2, dilation=1, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=2, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=4, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=8, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=16, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=32, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=64, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=128, bias=False),
        )

        self.judge = nn.Conv1d(channels, 1, 1, 1, 0, bias=False)

    def initialize_weights(self):
        for name, weight in self.named_parameters():
            if weight.data.dim() > 2:
                if 'judge' in name:
                    xavier_normal_(weight.data, 1)
                elif 'gate' in name:
                    xavier_normal_(
                        weight.data, calculate_gain('sigmoid'))
                else:
                    xavier_normal_(
                        weight.data, calculate_gain('tanh'))
        return self

    def forward(self, x):
        batch_size = x.shape[0]

        x = x.permute(0, 2, 1).contiguous().view(-1, 1, self.feature_channels)

        for i, layer in enumerate(self.encoder):
            if i == len(self.encoder) - 1:
                x = layer(x)
            else:
                x = F.leaky_relu(layer(x), 0.2)

        # x = self.ae[0].encode(x)

        # (batch * frames, latent, 1)
        x = x.view(batch_size, self.frames, self.channels)
        x = x.permute(0, 2, 1)

        features = []
        for layer, gate in zip(self.main, self.gate):
            p = layer.dilation[0]
            padded = F.pad(x, (p, 0))
            z = F.tanh(layer(padded)) * F.sigmoid(gate(padded))
            features.append(z)
            if x.shape[1] == z.shape[1]:
                x = z + x
            else:
                x = z

        latent = sum(features)
        x = self.judge(latent)

        x = x[:, :, -1:]
        return latent[:, :, -1:], x


class Discriminator(nn.Module):
    def __init__(self, frames, feature_channels, channels, n_judgements):
        super().__init__()
        self.n_judgements = n_judgements
        self.channels = channels
        self.feature_channels = feature_channels
        self.frames = frames

        self.spec = nn.Sequential(
            nn.Conv1d(feature_channels, channels, 3, 1, dilation=1, padding=1,
                      bias=False),
            nn.Conv1d(channels, channels, 3, 1, dilation=3, padding=3,
                      bias=False),
            nn.Conv1d(channels, channels, 3, 1, dilation=9, padding=9,
                      bias=False),
            nn.Conv1d(channels, channels, 3, 1, dilation=27, padding=27,
                      bias=False),
            nn.Conv1d(channels, channels, 3, 1, dilation=1, padding=1,
                      bias=False),
            nn.Conv1d(channels, channels, 3, 1, dilation=1, padding=1,
                      bias=False),
        )

        self.spec_judge = nn.Conv1d(channels, 1, 1, 1, 0, bias=False)

        self.loudness = nn.Sequential(
            nn.Conv1d(1, channels, 3, 1, dilation=1, padding=1, bias=False),
            nn.Conv1d(channels, channels, 3, 1, dilation=3, padding=3,
                      bias=False),
            nn.Conv1d(channels, channels, 3, 1, dilation=9, padding=9,
                      bias=False),
            nn.Conv1d(channels, channels, 3, 1, dilation=27, padding=27,
                      bias=False),
            nn.Conv1d(channels, channels, 3, 1, dilation=1, padding=1,
                      bias=False),
            nn.Conv1d(channels, channels, 3, 1, dilation=1, padding=1,
                      bias=False),
        )

        self.loudness_judge = nn.Conv1d(channels, 1, 1, 1, 0, bias=False)

        self.combo = nn.Sequential(
            nn.Conv1d(channels * 2, channels, 3, 1, dilation=1, padding=1,
                      bias=False),
            nn.Conv1d(channels, channels, 3, 1, dilation=3, padding=3,
                      bias=False),
            nn.Conv1d(channels, channels, 3, 1, dilation=9, padding=9,
                      bias=False),
            nn.Conv1d(channels, channels, 3, 1, dilation=27, padding=27,
                      bias=False),
            nn.Conv1d(channels, channels, 3, 1, dilation=1, padding=1,
                      bias=False),
            nn.Conv1d(channels, channels, 3, 1, dilation=1, padding=1,
                      bias=False),
        )
        self.combo_judge = nn.Conv1d(channels, 1, 1, 1, 0, bias=False)

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
        mx, encoded = x

        batch_size = encoded.shape[0]

        for layer in self.spec:
            encoded = F.leaky_relu(layer(encoded), 0.2)

        for layer in self.loudness:
            mx = F.leaky_relu(layer(mx), 0.2)

        spec_j = self.spec_judge(encoded)
        loud_j = self.loudness_judge(mx)

        c = torch.cat([encoded, mx], dim=1)
        for layer in self.combo:
            c = F.leaky_relu(layer(c), 0.2)

        combo_j = self.combo_judge(c)

        x = torch.cat([spec_j, loud_j, combo_j], dim=1)
        return x


class FrameDiscriminator(nn.Module):
    def __init__(self, frames, feature_channels, channels, n_judgements, ae):
        super().__init__()
        self.ae = [ae]
        self.n_judgements = n_judgements
        self.channels = channels
        self.feature_channels = feature_channels
        self.frames = frames

        self.stack = DownsamplingStack(
            start_size=feature_channels,
            target_size=1,
            scale_factor=2,
            layer_func=self._build_layer,
            activation=lambda x: F.leaky_relu(x, 0.2))
        self.judge = nn.Conv1d(channels, 1, 1, 1, 0, bias=False)

    def initialize_weights(self):
        for name, weight in self.named_parameters():
            if weight.data.dim() > 2:
                if 'judge' in name:
                    xavier_normal_(weight.data, 1)
                else:
                    xavier_normal_(
                        weight.data, calculate_gain('leaky_relu', 0.2))
        return self

    def _build_layer(self, i, curr_size, out_size, first, last):
        kernel_size = 7 if curr_size > 8 else 3
        padding = kernel_size // 2
        return nn.Conv1d(
            in_channels=1 if first else self.channels,
            out_channels=self.channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            bias=False)

    def forward(self, x):
        x = self.stack(x)
        x = self.judge(x)
        return x, x


class SpectrumDiscriminator(nn.Module):
    def __init__(self, feature_channels, channels, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.feature_channels = feature_channels
        self.spec_encoder = DownsamplingStack(
            start_size=feature_channels,
            target_size=1,
            scale_factor=2,
            layer_func=self._build_spec_layer,
            activation=lambda x: F.leaky_relu(x, 0.2))
        self.to_latent = nn.Conv1d(
            self.latent_dim, self.latent_dim, 1, 1, 0, bias=False)
        self.frame_judge = nn.Conv1d(self.latent_dim, 1, 1, 1, 0, bias=False)

    def _build_spec_layer(self, i, curr_size, out_size, first, last):
        kernel_size = 7 if curr_size > 8 else 3
        padding = kernel_size // 2
        return nn.Conv1d(
            in_channels=1 if first else self.channels,
            out_channels=self.latent_dim if last else self.channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            bias=False)

    def forward(self, x):
        batch_size = x.shape[0]
        x = flatten_channels(x, channels_as_time=True)
        x = self.spec_encoder(x)
        x = unflatten_channels(x, batch_size)
        x = self.to_latent(x)
        f_j = self.frame_judge(x).mean().view(1)
        return x, f_j


class TimeSeriesDiscriminator(nn.Module):
    def __init__(self, frames, features_channels, channels, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.features_channels = features_channels
        self.frames = frames

        self.time_encoder = DilatedStack(
            self.latent_dim,
            channels,
            3,
            [1, 3, 9, 1],
            activation=lambda x: F.leaky_relu(x, 0.2))

        self.time_crunch = DownsamplingStack(
            frames,
            1,
            2,
            layer_func=self._build_layer,
            activation=lambda x: F.leaky_relu(x, 0.2))

        # self.time_encoder = make_stack(
        #     frames,
        #     1,
        #     lambda i: nn.Conv1d(
        #         channels,
        #         channels,
        #         kernel_size=2,
        #         stride=2,
        #         padding=0,
        #         bias=False))
        # self.judges = nn.Sequential(*[nn.Conv1d(channels, 1, 1, 1, 0, bias=False) for _ in self.time_encoder])
        self.judge = nn.Conv1d(channels, 1, 1, 1, 0, bias=False)

    def _build_layer(self, i, curr_size, out_size, first, last):
        return nn.Conv1d(
            self.channels,
            self.channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

    def forward(self, x):
        batch_size = x.shape[0]
        # judgements = []
        # for layer, judge in zip(self.time_encoder, self.judges):
        #     x = F.leaky_relu(layer(x), 0.2)
        #     judgements.append(judge(x))

        x = self.time_encoder(x)
        for layer in self.time_crunch:
            x = F.leaky_relu(layer(x), 0.2)
        latent = x
        x = self.judge(x)
        # x = torch.cat([j.view(batch_size, -1) for j in judgements], dim=-1)
        return latent, x


class SpecDiscriminator(nn.Module):
    def __init__(self, frames, feature_channels, channels, n_judgements, ae):
        super().__init__()
        self.ae = [ae]
        self.n_judgements = n_judgements
        self.channels = channels
        self.feature_channels = feature_channels
        self.frames = frames

        self.latent_dim = 32


        self.spec_judge = SpectrumDiscriminator(
            feature_channels, channels, self.latent_dim)


        self.time_judge = TimeSeriesDiscriminator(
            frames, feature_channels, channels, self.latent_dim)

    def initialize_weights(self):
        for name, weight in self.named_parameters():
            if weight.data.dim() > 2:
                if 'judge' in name:
                    xavier_normal_(weight.data, 1)
                else:
                    xavier_normal_(
                        weight.data, calculate_gain('leaky_relu', 0.2))
        return self

    def _build_spec_layer(self, i, curr_size, out_size, first, last):
        kernel_size = 7 if curr_size > 8 else 3
        padding = kernel_size // 2
        return nn.Conv1d(
            in_channels=1 if first else self.channels,
            out_channels=self.latent_dim if last else self.channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            bias=False)

    def evaluate_latent(self, x):
        _, f_j = self.spec_judge(x)
        return f_j

    def forward(self, x):

        # spectral
        # batch_size = x.shape[0]
        # x = flatten_channels(x, channels_as_time=True)
        # x = self.spec_encoder(x)
        # x = unflatten_channels(x, batch_size)
        # x = self.to_latent(x)
        # f_j = self.frame_judge(x).mean().view(1)

        x, f_j = self.spec_judge(x)

        # time
        # x = self.time_encoder(x)
        latent = x
        # x = self.judge(x).mean().view(1)

        _, x = self.time_judge(x)
        return latent, x


class EnergyDiscriminator(nn.Module):
    """
    This discriminator judges whether loudness time series are real or fake
    """

    def __init__(self, frames, feature_channels, channels, n_judgements, ae):
        super().__init__()
        self.ae = [ae]
        self.n_judgements = n_judgements
        self.channels = channels
        self.feature_channels = feature_channels
        self.frames = frames

        self.loudness = nn.Sequential(
            nn.Conv1d(1, channels, 3, 1, dilation=1, padding=1, bias=False),
            nn.Conv1d(channels, channels, 3, 1, dilation=3, padding=3,
                      bias=False),
            nn.Conv1d(channels, channels, 3, 1, dilation=9, padding=9,
                      bias=False),
            nn.Conv1d(channels, channels, 3, 1, dilation=27, padding=27,
                      bias=False),
            nn.Conv1d(channels, channels, 3, 1, dilation=1, padding=1,
                      bias=False),
            nn.Conv1d(channels, channels, 3, 1, dilation=1, padding=1,
                      bias=False),
        )

        self.loudness_judge = nn.Conv1d(channels, 1, 1, 1, 0, bias=False)

    def initialize_weights(self):
        for name, weight in self.named_parameters():
            if weight.data.dim() > 2:
                # weight.data.normal_(0, 0.02)
                if 'judge' in name:
                    xavier_normal_(weight.data, 1)
                else:
                    xavier_normal_(
                        weight.data, calculate_gain('leaky_relu', 0.2))
        return self

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, self.frames)
        for layer in self.loudness:
            x = F.leaky_relu(layer(x), 0.2)
        x = self.loudness_judge(x)
        return x, x


class FullDiscriminator(nn.Module):
    def __init__(self, frames, feature_channels, channels, n_judgements, ae):
        super().__init__()
        self.ae = [ae]
        self.n_judgements = n_judgements
        self.channels = channels
        self.feature_channels = feature_channels
        self.frames = frames

        self.latent_dim = 32

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 128, 7, 2, 3, bias=False),
            nn.Conv1d(128, 128, 7, 2, 3, bias=False),
            nn.Conv1d(128, 128, 7, 2, 3, bias=False),
            nn.Conv1d(128, 128, 7, 2, 3, bias=False),
            nn.Conv1d(128, 128, 3, 2, 1, bias=False),
            nn.Conv1d(128, 128, 3, 2, 1, bias=False),
            nn.Conv1d(128, 128, 3, 2, 1, bias=False),
            nn.Conv1d(128, 128, 3, 2, 1, bias=False),
            nn.Conv1d(128, self.latent_dim, 1, 1, 0, bias=False),
        )

        self.frame_judge = nn.Conv1d(self.latent_dim, 1, 1, 1, 0, bias=False)

        # self.main = make_stack(
        #     frames,
        #     1,
        #     lambda i: nn.Conv1d(self.latent_dim if i == 0 else channels, channels, 7, 2, 3, bias=False))
        #
        #
        self.judge = nn.Conv1d(channels, 1, 1, 1, 0, bias=False)

        self.main = nn.Sequential(
            nn.Conv1d(self.latent_dim, channels, 2, 1, dilation=1, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=2, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=4, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=8, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=16, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=32, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=64, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=128, bias=False))

        # self.main = make_stack(frames, 1, lambda i: nn.Conv1d(self.latent_dim if i == 0 else channels, channels, 3, 2, 1, bias=False))
        # self.judges = make_stack(frames, 1, lambda i: nn.Conv1d(channels, 1, 3, 1, 1, bias=False))

    def initialize_weights(self):
        for name, weight in self.named_parameters():
            if weight.data.dim() > 2:
                # weight.data.normal_(0, 0.02)
                if 'judge' in name:
                    xavier_normal_(weight.data, 1)
                else:
                    xavier_normal_(
                        weight.data, calculate_gain('leaky_relu', 0.2))
        return self

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.feature_channels, -1)

        x = x.permute(0, 2, 1).contiguous().view(-1, 1, self.feature_channels)

        for i, layer in enumerate(self.encoder):
            if i == len(self.encoder) - 1:
                x = layer(x)
            else:
                x = F.leaky_relu(layer(x), 0.2)

        # x = self.ae[0].encode(x)

        # (batch * frames, latent, 1)
        x = x.view(batch_size, -1, self.latent_dim)
        latent = x = x.permute(0, 2, 1)

        f_j = self.frame_judge(x[:, :, -256:]).mean().view(1)
        # (batch, latent, frames)

        for layer in self.main:
            padded = F.pad(x, (layer.dilation[0], 0))
            z = layer(padded)
            if z.shape[1] == x.shape[1]:
                x = F.leaky_relu(x + z, 0.2)
            else:
                x = F.leaky_relu(z, 0.2)

        x = x[:, :, -256:]
        x = self.judge(x).mean().view(1)
        x = torch.cat([f_j, x])
        return latent, x
