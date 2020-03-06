import torch
from torch import nn
from torch.nn import functional as F


class ComplextSTFTGenerator(nn.Module):
    def __init__(self, input_channels, window_size, hop_size):
        super().__init__()
        self.hop_size = hop_size
        self.window_size = window_size
        self.input_channels = input_channels
        window = torch.hann_window(self.window_size).float()
        self.register_buffer('window', window)
        channels = window_size
        self.main = nn.Sequential(
            nn.Conv1d(self.input_channels, channels, 7, 1, 3),
            nn.Conv1d(channels, channels, 7, 1, 3),
            nn.Conv1d(channels, channels, 7, 1, 3),
            nn.Conv1d(channels, channels, 7, 1, 3),
            nn.Conv1d(channels, channels, 7, 1, 3))
        self.to_frames = nn.Conv1d(channels, channels + 2, 7, 1, 3)

    def forward(self, x):
        for layer in self.main:
            x = F.leaky_relu(layer(x), 0.2)
        x = self.to_frames(x)
        return x