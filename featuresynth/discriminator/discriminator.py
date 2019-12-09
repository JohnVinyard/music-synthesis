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
                nn.Conv1d(sl.stop - sl.start, channels, 1, 1, 0, bias=False))


        self.learned = nn.Sequential(
            nn.Conv1d(1, channels, 25, stride=8, padding=12, bias=False),
            nn.Conv1d(channels, channels, 25, stride=8, padding=12, bias=False),
        )
        self.unconditioned = nn.Conv1d(channels, channels, 1, 1, 0, bias=False)

        full = []
        n_layers = int(np.log2(input_size) - np.log2(feature_size))
        for i in range(n_layers):
            full.append(weight_norm(nn.Conv1d(
                filter_bank_channels + channels if i == 0 else channels,
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
            filter_bank_channels + channels, channels, 1, 1, 0, bias=False)))
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



    def forward(self, band, features):
        batch_size = features.shape[0]

        band = band.view(-1, 1, self.input_size)

        spectral = self.fb[0].convolve(band)[..., :self.input_size].contiguous()
        spectral = spectral * torch.abs(self.scale)
        spectral = F.relu(spectral)

        # frequency slice of incoming features
        features = features[:, self.sl, :]

        embedded = F.leaky_relu(self.feature_embedding(features), 0.2)

        features = []
        judgements = []

        x = band
        for layer in self.learned:
            x = F.leaky_relu(layer(x), 0.2)
            features.append(x)

        judgements.append(F.tanh(self.unconditioned(x)))


        # raw full spectrograms
        x = torch.cat(
            [spectral, F.upsample(embedded, size=spectral.shape[-1])], dim=1)
        for layer in self.full:
            x = F.leaky_relu(layer(x), 0.2)
            features.append(x)

        x = F.tanh(self.full_judge(x))
        judgements.append(x)

        # downsampled spectrograms
        kernel = spectral.shape[-1] // self.feature_size
        low_res = F.avg_pool1d(spectral, kernel, kernel)
        x = torch.cat([low_res, embedded], dim=1)
        for layer in self.main:
            x = F.leaky_relu(layer(x), 0.2)
            features.append(x)

        judgements.append(F.tanh(self.judge(x)))

        return \
            torch.cat([j.view(batch_size, -1) for j in judgements], dim=1), \
            features






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

    def forward(self, bands, features):
        batch_size = features.shape[0]
        features = features.view(-1, self.feature_channels, self.feature_size)

        # judgements = []
        # feat = []

        judgements = {}
        feat = defaultdict(list)

        for layer, band in zip(self.items, bands):
            j, f = layer(bands[band], features)
            size = bands[band].shape[-1]
            judgements[size] = j
            feat[size].extend(f)
            # judgements.append(j)
            # feat.extend(f)

        # return \
        #     torch.cat([j.view(batch_size, -1) for j in judgements], dim=1), \
        #     feat

        return judgements, feat
