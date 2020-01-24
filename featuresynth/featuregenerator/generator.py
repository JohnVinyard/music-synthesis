import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, calculate_gain, orthogonal_
import numpy as np
import math
from ..util.modules import \
    ToTimeSeries, UpSample, LearnedUpSample, UpsamplingStack, normalize, \
    flatten_channels, unflatten_channels, DilatedStack


# class CausalConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, dilation):
#         super().__init__()
#         self.out_channels = out_channels
#         self.in_channels = in_channels
#         self.padding = dilation * (kernel_size - 1)
#         self.conv = nn.Conv1d(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride=1,
#             padding=0,
#             dilation=dilation,
#             bias=False)
#         self.bn = nn.BatchNorm1d(out_channels)
#
#     def forward(self, x):
#         batch_size = x.shape[0]
#         x = inp = x.view(batch_size, self.in_channels, -1)
#         x = torch.cat([
#             torch.zeros(batch_size, self.in_channels, self.padding).to(
#                 x.device),
#             x
#         ], dim=-1)
#         x = self.conv(x)
#         if x.shape[1] == inp.shape[1]:
#             x = inp + x
#         x = F.leaky_relu(x, 0.2)
#         x = self.bn(x)
#         return x


def make_stack(start_size, target_size, layer_func):
    n_layers = int(np.log2(target_size) - np.log2(start_size))
    layers = []
    for i in range(n_layers):
        layers.append(layer_func(i))
    return nn.Sequential(*layers)


# class DilatedConvolution(nn.Conv1d):
#     def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1):
#         padding = dilation * (kernel_size - 1)
#         super().__init__(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride=1,
#             padding=0,
#             dilation=dilation,
#             bias=False)



class ARGenerator(nn.Module):
    def __init__(self, frames, out_channels, noise_dim, initial_dim, channels,
                 ae):
        super().__init__()
        self.initial_dim = initial_dim
        self.out_channels = out_channels
        self.noise_dim = noise_dim
        self.channels = channels
        self.frames = frames

        self.main = nn.Sequential(
            nn.Conv1d(out_channels, channels, 2, dilation=1, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=2, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=4, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=8, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=16, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=32, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=64, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=128, bias=False),
        )

        self.gate = nn.Sequential(
            nn.Conv1d(out_channels, channels, 2, dilation=1, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=2, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=4, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=8, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=16, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=32, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=64, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=128, bias=False),
        )

        # self.decoder = nn.Sequential(
        #     nn.Conv1d(channels, 256, 1, 1, 0, bias=False),
        #     nn.ConvTranspose1d(128, 128, 4, 2, 1, bias=False),
        #     nn.ConvTranspose1d(128, 128, 4, 2, 1, bias=False),
        #     nn.ConvTranspose1d(128, 128, 4, 2, 1, bias=False),
        #     nn.ConvTranspose1d(128, 128, 8, 2, 3, bias=False),
        #     nn.ConvTranspose1d(128, 128, 8, 2, 3, bias=False),
        #     nn.ConvTranspose1d(128, 128, 8, 2, 3, bias=False),
        #     nn.ConvTranspose1d(128, 1, 8, 2, 3, bias=False),
        # )

        self.decoder = nn.Sequential(
            nn.Conv1d(channels, 256, 1, 1, 0, bias=False),
            nn.Conv1d(128, 128, 3, 1, 1, bias=False),
            nn.Conv1d(128, 128, 3, 1, 1, bias=False),
            nn.Conv1d(128, 128, 3, 1, 1, bias=False),
            nn.Conv1d(128, 128, 7, 1, 3, bias=False),
            nn.Conv1d(128, 128, 7, 1, 3, bias=False),
            nn.Conv1d(128, 128, 7, 1, 3, bias=False),
            nn.Conv1d(128, 1, 7, 1, 3, bias=False),
        )

        self.to_frames = nn.Conv1d(channels, out_channels, 1, 1, 0, bias=False)

    def initialize_weights(self):
        for name, weight in self.named_parameters():
            if weight.data.dim() > 2:
                if 'to_frames' in name:
                    xavier_normal_(weight.data, 1)
                elif 'gate' in name:
                    xavier_normal_(weight.data, calculate_gain('sigmoid'))
                elif 'decoder' in name:
                    xavier_normal_(weight.data,
                                   calculate_gain('leaky_relu', 0.2))
                else:
                    xavier_normal_(weight.data, calculate_gain('tanh'))
        return self

    def generate(self, primer, steps):
        with torch.no_grad():
            x = primer.view(1, self.out_channels, self.frames)
            generated = []
            for i in range(steps):
                new_frame = self.forward(x)[1][:, :, -1:]
                # print(new_frame.shape, new_frame.std().item())
                generated.append(new_frame)
                x = torch.cat([x, new_frame], dim=-1)[..., 1:]
            result = torch.cat(generated, dim=-1)
            # print(result.shape)
            return result

    def forward(self, x):
        batch_size = x.shape[0]
        orig = x = x.view(batch_size, self.out_channels, self.frames)

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

        x = sum(features)

        x = x.permute(0, 2, 1).contiguous().view(-1, self.channels, 1)

        for i, layer in enumerate(self.decoder):
            if i == 0:
                # (batch * frames, latent_dim, 1)
                x = F.leaky_relu(layer(x), 0.2)
                # (batch * frames, latent_dim * 2, 1)
                x = x.permute(0, 2, 1).contiguous().view(
                    batch_size * self.frames, 2, 128).permute(0, 2,
                                                              1).contiguous()
                # x = x.view(batch_size * self.frames, -1, 2)
                # x = self.decoder_bns[i](x)
            elif i == len(self.decoder) - 1:
                x = F.upsample(x, scale_factor=2)
                x = layer(x) ** 2
            else:
                x = F.upsample(x, scale_factor=2)
                x = F.leaky_relu(layer(x), 0.2)
                # x = self.decoder_bns[i](x)

        # x = self.ae[0].decode(x)

        # (batch * frames, 1, feature_channels)
        x = x.view(batch_size, self.frames, self.out_channels) \
            .permute(0, 2, 1).contiguous()

        x = torch.cat([orig[:, :, :-1], x[:, :, -1:]], dim=-1)
        return x, x


class FrameGenerator(nn.Module):
    def __init__(self, frames, out_channels, noise_dim, initial_dim, channels,
                 ae):
        super().__init__()
        self.ae = [ae]
        self.initial_dim = initial_dim
        self.out_channels = out_channels
        self.noise_dim = noise_dim
        self.channels = channels
        self.frames = frames

        self.to_time_series = ToTimeSeries(noise_dim, channels, initial_dim)
        self.stack = UpsamplingStack(
            initial_dim,
            out_channels,
            scale_factor=2,
            layer_func=self._build_layer)

    def initialize_weights(self):
        for name, weight in self.named_parameters():
            if weight.data.dim() > 2:
                xavier_normal_(weight.data, calculate_gain('leaky_relu', 0.2))
        return self

    def _build_layer(self, i, curr_size, out_size, first, last):
        return UpSample(
            in_channels=self.channels,
            out_channels=1 if last else self.channels,
            kernel_size=7,
            scale_factor=2,
            activation=self._to_frames if last else self._relu)

    def _relu(self, x):
        return F.leaky_relu(x, 0.2)

    def _to_frames(self, x):
        return normalize(x ** 2)

    def forward(self, x):
        x = F.leaky_relu(self.to_time_series(x), 0.2)
        x = self.stack(x)
        return x, x


class TimeSeriesGenerator(nn.Module):
    """
    Transform a noise vector into a time-series of latent frames
    """

    def __init__(
            self,
            frames,
            latent_out_channels,
            noise_dim,
            initial_dim,
            channels):
        super().__init__()
        self.channels = channels
        self.initial_dim = initial_dim
        self.noise_dim = noise_dim
        self.latent_out_channels = latent_out_channels
        self.frames = frames

        self.to_time_series = ToTimeSeries(noise_dim, channels, initial_dim)
        self.initial = DilatedStack(
            channels, channels, 3, [1, 3], lambda x: F.leaky_relu(x, 0.2),
            residual=False)
        self.final = DilatedStack(
            channels,
            channels,
            3,
            [27, 9, 3, 1, 1, 1],
            activation=lambda x: F.leaky_relu(x, 0.2), residual=False)
        self.to_latent = nn.Conv1d(
            channels, self.latent_out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        x = self.to_time_series(x)
        x = self.initial(x)
        x = F.upsample(x, size=self.frames)
        x = self.final(x)
        x = self.to_latent(x)
        return x


class SpectrumGenerator(nn.Module):
    def __init__(self, latent_dim, channels, initial_dim, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.initial_dim = initial_dim
        self.channels = channels
        self.latent_dim = latent_dim

        self.to_spec_series = ToTimeSeries(
            self.latent_dim, channels, initial_dim)
        self.spec_stack = UpsamplingStack(
            initial_dim,
            out_channels,
            scale_factor=2,
            layer_func=self._build_spec_layer)

    def _build_spec_layer(self, i, curr_size, out_size, first, last):
        return UpSample(
            in_channels=self.channels,
            out_channels=1 if last else self.channels,
            kernel_size=7,
            scale_factor=2,
            activation=self._to_frames if last else self._relu)

    def _relu(self, x):
        return F.leaky_relu(x, 0.2)

    def _to_frames(self, x):
        return normalize(x ** 2)

    def forward(self, x):
        batch_size = x.shape[0]
        x = flatten_channels(x)
        x = self.to_spec_series(x)
        x = self.spec_stack(x)
        x = unflatten_channels(x, batch_size)
        return x


class SpecGenerator(nn.Module):
    def __init__(
            self, frames, out_channels, noise_dim, initial_dim, channels, ae):

        super().__init__()
        self.ae = [ae]
        self.initial_dim = initial_dim
        self.out_channels = out_channels
        self.noise_dim = noise_dim
        self.channels = channels
        self.frames = frames

        self.latent_dim = 32

        self.to_time_series = TimeSeriesGenerator(
            frames, self.latent_dim, noise_dim, initial_dim, channels)

        self.spec = SpectrumGenerator(
            self.latent_dim, channels, initial_dim, out_channels)

    def initialize_weights(self):
        for name, weight in self.named_parameters():
            if weight.data.dim() > 2:
                xavier_normal_(weight.data, calculate_gain('leaky_relu', 0.2))
        return self

    def forward(self, x):
        latent = x = self.to_time_series(x)
        x = self.spec(x)
        return latent, x


class EnergyGenerator(nn.Module):
    """
    This generator creates loudness time series
    """

    def __init__(self, frames, out_channels, noise_dim, initial_dim, channels,
                 ae):
        super().__init__()
        self.ae = [ae]
        self.initial_dim = initial_dim
        self.out_channels = out_channels
        self.noise_dim = noise_dim
        self.channels = channels
        self.frames = frames

        self.to_time_series = nn.Linear(
            noise_dim, initial_dim * channels, bias=False)

        self.initial = nn.Sequential(
            nn.Conv1d(channels, channels, 3, 1, dilation=1, padding=1,
                      bias=False),
            nn.Conv1d(channels, channels, 3, 1, dilation=3, padding=3,
                      bias=False),
        )

        c = channels

        self.loudness = nn.Sequential(
            nn.Conv1d(c, c, 3, 1, dilation=27, padding=27, bias=False),
            nn.Conv1d(c, c, 3, 1, dilation=9, padding=9, bias=False),
            nn.Conv1d(c, c, 3, 1, dilation=3, padding=3, bias=False),
            nn.Conv1d(c, c, 3, 1, dilation=1, padding=1, bias=False),
            nn.Conv1d(c, c, 3, 1, dilation=1, padding=1, bias=False),
            nn.Conv1d(c, c, 3, 1, dilation=1, padding=1, bias=False),
        )

        self.to_frames = nn.Conv1d(channels, 1, 7, 1, 3, bias=False)

    def initialize_weights(self):
        for name, weight in self.named_parameters():
            if weight.data.dim() > 2:
                if 'to_frames' in name:
                    xavier_normal_(weight.data, 1)
                else:
                    xavier_normal_(weight.data,
                                   calculate_gain('leaky_relu', 0.2))
        return self

    def normalize(self, l, scaling=1):
        batch_size = l.shape[0]
        orig_shape = l.shape
        l = l.view(batch_size, -1)
        mx, _ = torch.abs(l).max(dim=1, keepdim=True)
        l = l / (mx + 1e-8)
        l = l.view(*orig_shape)
        return l * scaling

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = F.leaky_relu(self.to_time_series(x), 0.2)
        x = x.view(batch_size, self.channels, self.initial_dim)

        for layer in self.initial:
            z = layer(x)
            x = F.leaky_relu(x + z, 0.2)

        x = F.upsample(x, size=self.frames)

        for layer in self.loudness:
            z = layer(x)
            x = F.leaky_relu(x + z, 0.2)

        # x = self.to_frames(x) ** 2
        # x = self.normalize(x)
        x = F.sigmoid(self.to_frames(x))
        return x, x


class FullGenerator(nn.Module):
    def __init__(self, frames, out_channels, noise_dim, initial_dim, channels,
                 ae):
        super().__init__()
        self.ae = [ae]
        self.initial_dim = initial_dim
        self.out_channels = out_channels
        self.noise_dim = noise_dim
        self.channels = channels
        self.frames = frames

        self.latent_dim = 32

        self.to_time_series = nn.Linear(
            self.noise_dim, initial_dim * self.channels, bias=False)
        self.upsample = make_stack(
            initial_dim,
            frames,
            lambda i: nn.ConvTranspose1d(channels, channels, 4, 2, 1,
                                         bias=False))

        self.transform = nn.Sequential(
            nn.Conv1d(channels, channels, 2, 1, dilation=1, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=2, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=4, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=8, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=16, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=32, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=64, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=128, bias=False),
        )

        self.to_latent = nn.Conv1d(channels, self.latent_dim, 1, 1, 0,
                                   bias=False)

        self.decoder = nn.Sequential(
            nn.Conv1d(self.latent_dim, 256, 1, 1, 0, bias=False),
            nn.ConvTranspose1d(128, 128, 4, 2, 1, bias=False),
            nn.ConvTranspose1d(128, 128, 4, 2, 1, bias=False),
            nn.ConvTranspose1d(128, 128, 4, 2, 1, bias=False),
            nn.ConvTranspose1d(128, 128, 8, 2, 3, bias=False),
            nn.ConvTranspose1d(128, 128, 8, 2, 3, bias=False),
            nn.ConvTranspose1d(128, 128, 8, 2, 3, bias=False),
            nn.ConvTranspose1d(128, 1, 8, 2, 3, bias=False),
        )


        # self.decoder = nn.Sequential(
        #     nn.Conv1d(self.latent_dim, 256, 1, 1, 0, bias=False),
        #     nn.Conv1d(128, 128, 3, 1, 1, bias=False),
        #     nn.Conv1d(128, 128, 3, 1, 1, bias=False),
        #     nn.Conv1d(128, 128, 3, 1, 1, bias=False),
        #     nn.Conv1d(128, 128, 7, 1, 3, bias=False),
        #     nn.Conv1d(128, 128, 7, 1, 3, bias=False),
        #     nn.Conv1d(128, 128, 7, 1, 3, bias=False),
        #     nn.Conv1d(128, 128, 7, 1, 3, bias=False),
        #     nn.Conv1d(128, 1, 7, 1, 3, bias=False),
        # )

    def initialize_weights(self):
        for name, weight in self.named_parameters():
            if weight.data.dim() > 2:
                # weight.data.normal_(0, 0.02)
                if 'to_loudness_frames' in name:
                    xavier_normal_(weight.data, calculate_gain('sigmoid'))
                if 'to_frames' in name:
                    xavier_normal_(weight.data, 1)
                else:
                    xavier_normal_(weight.data,
                                   calculate_gain('leaky_relu', 0.2))
        return self

    def normalize(self, l, scaling=1):
        batch_size = l.shape[0]
        orig_shape = l.shape
        l = l.view(batch_size, -1)
        mx, _ = torch.abs(l).max(dim=1, keepdim=True)
        l = l / (mx + 1e-8)
        l = l.view(*orig_shape)
        return l * scaling

    def forward(self, x):
        batch_size = x.shape[0]

        # x = x.view(batch_size, -1)
        # x = F.leaky_relu(self.to_time_series(x), 0.2)
        # x = x.view(batch_size, self.channels, self.initial_dim)
        #
        # for i, layer in enumerate(self.upsample):
        #     x = F.leaky_relu(layer(x), 0.2)

        for layer in self.transform:
            padded = F.pad(x, (layer.dilation[0], 0))
            z = layer(padded)
            x = F.leaky_relu(x + z, 0.2)

        x = self.to_latent(x)
        latent = x

        x = x.permute(0, 2, 1).contiguous().view(-1, self.latent_dim, 1)

        for i, layer in enumerate(self.decoder):
            if i == 0:
                # (batch * frames, latent_dim, 1)
                x = layer(x)
                # (batch * frames, latent_dim * 2, 1)
                x = x.view(-1, 128, 2)
                # x = x.permute(0, 2, 1).contiguous().view(
                #     batch_size * self.frames, 2, 128).permute(0, 2, 1).contiguous()
            else:
                # x = F.upsample(x, scale_factor=2)
                x = layer(x)

            if i == len(self.decoder) - 1:
                x = x ** 2
            else:
                x = F.leaky_relu(x, 0.2)

        # (batch * frames, 1, feature_channels)
        x = x.view(batch_size, -1, self.out_channels) \
            .permute(0, 2, 1).contiguous()

        x = self.normalize(x)
        return latent, x


class Generator(nn.Module):
    def __init__(self, frames, out_channels, noise_dim, initial_dim, channels):
        super().__init__()
        self.initial_dim = initial_dim
        self.out_channels = out_channels
        self.noise_dim = noise_dim
        self.channels = channels
        self.frames = frames

        self.to_loudness_time_series = nn.Linear(
            self.noise_dim, initial_dim * self.channels)

        self.to_spec_time_series = nn.Linear(
            self.noise_dim, initial_dim * self.channels)

        self.loudness_initial = nn.Sequential(
            nn.Conv1d(channels, channels, 3, 1, dilation=1, padding=1,
                      bias=False),
            nn.Conv1d(channels, channels, 3, 1, dilation=3, padding=3,
                      bias=False),
        )

        self.spec_initial = nn.Sequential(
            nn.Conv1d(channels, channels, 3, 1, dilation=1, padding=1,
                      bias=False),
            nn.Conv1d(channels, channels, 3, 1, dilation=3, padding=3,
                      bias=False),
        )

        c = channels

        self.loudness = nn.Sequential(
            nn.Conv1d(c, c, 3, 1, dilation=27, padding=27, bias=False),
            nn.Conv1d(c, c, 3, 1, dilation=9, padding=9, bias=False),
            nn.Conv1d(c, c, 3, 1, dilation=3, padding=3, bias=False),
            nn.Conv1d(c, c, 3, 1, dilation=1, padding=1, bias=False),
            nn.Conv1d(c, c, 3, 1, dilation=1, padding=1, bias=False),
            nn.Conv1d(c, c, 3, 1, dilation=1, padding=1, bias=False),
        )

        self.spec = nn.Sequential(
            nn.Conv1d(c, c, 3, 1, dilation=27, padding=27, bias=False),
            nn.Conv1d(c, c, 3, 1, dilation=9, padding=9, bias=False),
            nn.Conv1d(c, c, 3, 1, dilation=3, padding=3, bias=False),
            nn.Conv1d(c, c, 3, 1, dilation=1, padding=1, bias=False),
            nn.Conv1d(c, c, 3, 1, dilation=1, padding=1, bias=False),
            nn.Conv1d(c, c, 3, 1, dilation=1, padding=1, bias=False),
        )

        self.to_frames = nn.Conv1d(
            channels,
            out_channels,
            1,
            1,
            0)

        self.to_loudness_frames = nn.Conv1d(
            channels,
            1,
            7,
            1,
            3,
            bias=False)

    def initialize_weights(self):
        for name, weight in self.named_parameters():
            if weight.data.dim() > 2:
                if 'to_loudness_frames' in name:
                    xavier_normal_(weight.data, calculate_gain('sigmoid'))
                if 'to_frames' in name:
                    xavier_normal_(weight.data, 1)
                else:
                    xavier_normal_(weight.data,
                                   calculate_gain('leaky_relu', 0.2))
        return self

    def normalize(self, l, scaling=1):
        batch_size = l.shape[0]
        orig_shape = l.shape
        l = l.view(batch_size, -1)
        mx, _ = torch.abs(l).max(dim=1, keepdim=True)
        l = l / (mx + 1e-8)
        l = l.view(*orig_shape)
        return l * scaling

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.noise_dim)
        s = l = x

        s = F.leaky_relu(self.to_spec_time_series(x), 0.2)
        s = s.view(batch_size, self.channels, self.initial_dim)

        l = F.leaky_relu(self.to_loudness_time_series(x), 0.2)
        l = l.view(batch_size, self.channels, self.initial_dim)

        for layer in self.spec_initial:
            z = layer(s)
            s = F.leaky_relu(s + z, 0.2)

        for layer in self.loudness_initial:
            z = layer(l)
            l = F.leaky_relu(l + z, 0.2)

        s = F.upsample(s, size=self.frames)
        l = F.upsample(l, size=self.frames)

        for layer in self.spec:
            z = layer(s)
            s = F.leaky_relu(s + z, 0.2)

        for layer in self.loudness:
            z = layer(l)
            l = F.leaky_relu(l + z, 0.2)

        l = F.sigmoid(self.to_loudness_frames(l))
        s = self.to_frames(s)
        # s = self.normalize(s, scaling=1.5)
        return l, s


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

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

        self.decoder = nn.Sequential(
            nn.Conv1d(self.latent_dim, 256, 1, 1, 0, bias=False),
            nn.ConvTranspose1d(128, 128, 4, 2, 1, bias=False),
            nn.ConvTranspose1d(128, 128, 4, 2, 1, bias=False),
            nn.ConvTranspose1d(128, 128, 4, 2, 1, bias=False),
            nn.ConvTranspose1d(128, 128, 8, 2, 3, bias=False),
            nn.ConvTranspose1d(128, 128, 8, 2, 3, bias=False),
            nn.ConvTranspose1d(128, 128, 8, 2, 3, bias=False),
            nn.ConvTranspose1d(128, 1, 8, 2, 3, bias=False),
        )

    def initialize_weights(self):
        for name, weight in self.named_parameters():
            if weight.data.dim() > 2:
                if 'to_frames' in name:
                    orthogonal_(weight.data, 1)
                else:
                    orthogonal_(weight.data, calculate_gain('leaky_relu', 0.2))
        return self

    def encode(self, x):
        for i, layer in enumerate(self.encoder):
            if i == len(self.encoder) - 1:
                x = layer(x)
            else:
                x = F.leaky_relu(layer(x), 0.2)

        encoded = x
        return encoded

    def decode(self, x):
        batch_size = x.shape[0]

        for i, layer in enumerate(self.decoder):
            if i == 0:
                x = F.leaky_relu(layer(x), 0.2)
                x = x.view(batch_size, -1, 2)
            elif i == len(self.decoder) - 1:
                x = layer(x) ** 2
            else:
                x = F.leaky_relu(layer(x), 0.2)
        return x

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, 256)

        encoded = self.encode(x)
        x = self.decode(encoded)

        return encoded, x
