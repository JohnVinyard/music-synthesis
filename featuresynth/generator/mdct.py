from torch import nn
from torch.nn.init import xavier_normal_, calculate_gain
from ..util.modules import DilatedStack, DownsamplingStack, UpsamplingStack
from torch.nn import functional as F
import torch


class MDCTGenerator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        channels = 1024
        self.main = DilatedStack(
            in_channels,
            channels,
            3,
            [1, 3, 9, 1, 1],
            groups=[1, 4, 16, 64, 128],
            activation=lambda x: F.leaky_relu(x, 0.2),
            residual=True)

        self.to_frames = nn.Conv1d(channels, 256, 7, 1, 3, groups=256)
        self.to_frames_gate = nn.Conv1d(channels, 256, 7, 1, 3, groups=256)

    # def initialize_weights(self):
    #     for name, weight in self.named_parameters():
    #         if weight.data.dim() > 2:
    #             if 'frames' in name:
    #                 xavier_normal_(weight.data, calculate_gain('tanh'))
    #             else:
    #                 xavier_normal_(
    #                     weight.data, calculate_gain('leaky_relu', 0.2))
    #     return self

    def forward(self, x):
        batch, channels, time = x.shape
        x = self.main(x)
        x = F.tanh(self.to_frames(x)) * F.tanh(self.to_frames_gate(x))
        x = x.view(batch, -1, time)
        return x


class GroupedMDCTGenerator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, 512, 7, 1, 3),
            nn.Conv1d(512, 1024, 7, 1, 3, groups=4),
            nn.Conv1d(1024, 2048, 7, 1, 3, groups=16),
            nn.Conv1d(2048, 4096, 7, 1, 3, groups=64),
        )

        self.to_frames = nn.Conv1d(4096, 256, 7, 1, 3, groups=256)
        self.to_frames_gate = nn.Conv1d(4096, 256, 7, 1, 3, groups=256)

    # def initialize_weights(self):
    #     for name, weight in self.named_parameters():
    #         if weight.data.dim() > 2:
    #             if 'frames' in name:
    #                 xavier_normal_(weight.data, calculate_gain('tanh'))
    #             else:
    #                 xavier_normal_(
    #                     weight.data, calculate_gain('leaky_relu', 0.2))
    #     return self

    def forward(self, x):
        batch, channels, time = x.shape
        for layer in self.main:
            x = F.leaky_relu(layer(x), 0.2)
        x = F.tanh(self.to_frames(x)) * F.tanh(self.to_frames_gate(x))
        x = x.view(batch, -1, time)
        return x


class UnconditionedGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise_dim = 128
        self.initial = nn.Linear(self.noise_dim, 4 * 4 * 1024, bias=False)
        self.stack = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.ConvTranspose2d(512, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.ConvTranspose2d(256, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.ConvTranspose2d(128, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.ConvTranspose2d(128, 64, (4, 3), (2, 1), (1, 1), bias=False),

        )
        self.to_frames = nn.ConvTranspose2d(
            64, 1, (4, 3), (2, 1), (1, 1), bias=False)
        self.to_frames_gate = nn.ConvTranspose2d(
            64, 1, (4, 3), (2, 1), (1, 1), bias=False)

    def initialize_weights(self):
        for name, weight in self.named_parameters():
            if weight.data.dim() > 2:
                if 'to_frames' in name:
                    xavier_normal_(weight.data, calculate_gain('tanh'))
                else:
                    xavier_normal_(weight.data,
                                   calculate_gain('leaky_relu', 0.2))
        return self

    def forward(self, x):
        x = torch.FloatTensor(
            x.shape[0], self.noise_dim).normal_(0, 1).to(x.device)
        x = F.leaky_relu(self.initial(x), 0.2)
        x = x.view(x.shape[0], -1, 4, 4)
        for layer in self.stack:
            x = F.leaky_relu(layer(x), 0.2)
        x = F.tanh(self.to_frames(x)) * F.tanh(self.to_frames_gate(x))
        x = x.view(x.shape[0], 256, 64)
        return x


class TwoDimMDCTGenerator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.project = nn.Sequential(
            nn.Conv1d(in_channels, 512, 7, 1, 3),
            nn.Conv1d(512, 1024, 7, 1, 3),
            nn.Conv1d(1024, 2048, 7, 1, 3),
            nn.Conv1d(2048, 2048, 7, 1, 3)
        )

        # self.bootstrap = nn.Conv1d(256, 256 * 8, 1, 1, 0)

        self.refine = nn.Sequential(
            nn.ConvTranspose2d(256, 128, (4, 3), (2, 1), (1, 1)),
            nn.ConvTranspose2d(128, 64, (4, 3), (2, 1), (1, 1)),
            nn.ConvTranspose2d(64, 32, (4, 3), (2, 1), (1, 1)),
            nn.ConvTranspose2d(32, 16, (4, 3), (2, 1), (1, 1)),
        )

        self.to_frames = nn.ConvTranspose2d(16, 1, (4, 3), (2, 1), (1, 1))
        self.to_frames_gate = nn.ConvTranspose2d(16, 1, (4, 3), (2, 1), (1, 1))

    def initialize_weights(self):
        for name, weight in self.named_parameters():
            if weight.data.dim() > 2:
                if 'frames' in name:
                    xavier_normal_(weight.data, calculate_gain('tanh'))
                else:
                    xavier_normal_(
                        weight.data, calculate_gain('leaky_relu', 0.2))
        return self

    def forward(self, x):
        batch, channels, time = x.shape

        for layer in self.project:
            x = F.leaky_relu(layer(x), 0.2)

        # x = F.leaky_relu(self.bootstrap(x), 0.2)
        x = x.view(batch, -1, 8, time)

        for layer in self.refine:
            x = F.leaky_relu(layer(x), 0.2)

        x = F.tanh(self.to_frames(x)) * F.tanh(self.to_frames_gate(x))
        x = x.view(batch, channels, time)
        return x
