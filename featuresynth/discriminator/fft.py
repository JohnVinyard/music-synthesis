import torch
from torch import nn
from torch.nn import functional as F


class ComplextSTFTDiscriminator(nn.Module):
    def __init__(self, window_size, hop):
        super().__init__()
        self.hop = hop
        self.window_size = window_size
        window = torch.hann_window(self.window_size).float()
        self.register_buffer('window', window)
        channels = 512
        self.main = nn.Sequential(
            nn.Conv1d(window_size + 2, channels, 7, 2, 3),
            nn.Conv1d(channels, channels, 7, 2, 3),
            nn.Conv1d(channels, channels, 7, 2, 3),
            nn.Conv1d(channels, channels, 7, 2, 3),
        )
        self.judge = nn.Conv1d(channels, 1, 7, 1, 3)

    def forward(self, x):
        features = []
        for layer in self.main:
            x = F.leaky_relu(layer(x), 0.2)
            features.append(x)
        j = self.judge(x)
        return [features], [j]
