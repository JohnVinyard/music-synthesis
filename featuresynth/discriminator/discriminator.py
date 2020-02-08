import zounds
from torch import nn
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, calculate_gain
from torch.nn.utils import weight_norm
from collections import defaultdict


def weight_norm(x):
    return x




class LowResChannelJudge(nn.Module):
    def __init__(
            self,
            input_size,
            channels,
            feature_size,
            feature_channels,
            fb,
            sl):

        super().__init__()
        self.sl = sl
        self.feature_size = feature_size
        self.feature_channels = feature_channels
        self.channels = channels
        self.input_size = input_size

        filter_bank_channels = fb.filter_bank.shape[0]
        self.filter_bank_channels = filter_bank_channels

        self.feature_embedding = \
            weight_norm(
                nn.Conv1d(feature_channels, channels, 1, 1, 0, bias=False))


        self.learned = nn.Sequential(
            nn.Conv1d(1, channels, 25, stride=8, padding=12, bias=False),
            nn.Conv1d(channels, channels, 25, stride=8, padding=12, bias=False),
        )
        self.unconditioned = nn.Conv1d(channels, channels, 1, 1, 0, bias=False)

        full = []
        n_layers = int(np.log2(input_size) - np.log2(feature_size))
        for i in range(n_layers):
            full.append(weight_norm(nn.Conv1d(
                filter_bank_channels if i == 0 else channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False)))
        self.full = nn.Sequential(*full)
        self.full_judge = weight_norm(
            nn.Conv1d(channels, 1, 3, 2, 1, bias=False))


        layers = []
        layers.append(weight_norm(nn.Conv1d(
            filter_bank_channels, channels, 1, 1, 0, bias=False)))
        for i in range(int(np.log2(feature_size))):
            layers.append(weight_norm(nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False)))

        self.main = nn.Sequential(*layers)
        self.judge = weight_norm(
            nn.Conv1d(channels, 1, 1, 1, padding=0, bias=False))
        self.scale = nn.Parameter(torch.FloatTensor(1).fill_(0.0001))
        self.fb = [fb]



    def forward(self, band):
        batch_size = band.shape[0]

        band = band.view(-1, 1, self.input_size)

        spectral = self.fb[0].convolve(band)[..., :self.input_size].contiguous()
        spectral = spectral * torch.abs(self.scale)
        spectral = F.relu(spectral)


        features = []
        judgements = []

        # unconditioned
        x = band
        for layer in self.learned:
            x = F.leaky_relu(layer(x), 0.2)
            features.append(x)
        unconditioned = x
        judgements.append(F.tanh(self.unconditioned(x)))

        # raw, full spectorgams
        x = spectral
        for layer in self.full:
            x = F.leaky_relu(layer(x), 0.2)
            features.append(x)
        full_spec = x
        x = F.tanh(self.full_judge(x))
        judgements.append(x)

        # downsampled spectrograms
        kernel = spectral.shape[-1] // self.feature_size
        low_res = F.avg_pool1d(spectral, kernel, kernel)
        x = low_res
        for layer in self.main:
            x = F.leaky_relu(layer(x), 0.2)
            features.append(x)
        low_res = x

        judgements.append(F.tanh(self.judge(x)))
        j = torch.cat([j.view(batch_size, -1) for j in judgements], dim=1)
        return j, features, unconditioned, full_spec, low_res


class Discriminator(nn.Module):
    def __init__(
            self,
            input_sizes,
            feature_size,
            feature_channels,
            channels,
            kernel_size,
            filter_banks,
            slices):

        super().__init__()
        self.slices = slices
        self.kernel_size = kernel_size
        self.channels = channels
        self.feature_channels = feature_channels
        self.feature_size = feature_size
        self.input_sizes = input_sizes
        self.nframes = self.feature_size

        self.items = nn.Sequential(
            *[LowResChannelJudge(size, channels, feature_size, feature_channels,
                                 fb, sl)
              for size, fb, sl in zip(input_sizes, filter_banks, slices)])

        self.unconditioned_judge = nn.Conv1d(channels * 5, 1, 3, 1, 1, bias=False)
        self.full_spec_judge = nn.Conv1d(channels * 5, 1, 3, 1, 1, bias=False)
        self.low_res_judge = nn.Conv1d(channels * 5, 1, 1, 1, 0, bias=False)

    def initialize_weights(self):
        for name, weight in self.named_parameters():
            if weight.data.dim() > 2:
                if 'judge' in name:
                    xavier_normal_(weight.data, calculate_gain('tanh'))
                else:
                    xavier_normal_(
                        weight.data, calculate_gain('leaky_relu', 0.2))
        return self

    def test(self):
        batch_size = 8
        bands = [
            torch.FloatTensor(*(batch_size, 1, size))
            for size in self.input_sizes]
        features = torch.FloatTensor(
            *(batch_size, self.feature_channels, self.feature_size))
        out = self.forward(bands, features)
        print(out.shape)

    def forward(self, bands):
        batch_size = list(bands.values())[0].shape[0]

        judgements = []
        feat = []

        unconditioned = []
        full_spec = []
        low_res = []

        for layer, band in zip(self.items, bands):
            j, f, un, full, low = layer(bands[band])
            unconditioned.append(F.avg_pool1d(un, un.shape[-1] // 16))
            full_spec.append(full)
            low_res.append(low)
            judgements.append(j)
            feat.extend(f)

        unconditioned = torch.cat(unconditioned, dim=1)
        u = F.tanh(self.unconditioned_judge(unconditioned))
        judgements.append(u)

        full_spec = torch.cat(full_spec, dim=1)
        fs = F.tanh(self.full_spec_judge(full_spec))
        judgements.append(fs)

        lr = torch.cat(low_res, dim=1)
        lr = F.tanh(self.low_res_judge(lr))
        judgements.append(lr)

        judgements = torch.cat(
            [j.view(batch_size, -1) for j in judgements], dim=-1)
        return judgements, feat


