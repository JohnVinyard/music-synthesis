from torch import nn
from torch.nn.init import xavier_normal_, calculate_gain
import torch
from ..util.modules import DilatedStack
from torch.nn import functional as F


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
            activation=lambda x: F.leaky_relu(x, 0.2),
            residual=True)
        self.to_frames = nn.Conv1d(channels, 256, 3, 1, 1)

    def initialize_weights(self):
        for name, weight in self.named_parameters():
            if weight.data.dim() > 2:
                if 'frames' in name:
                    xavier_normal_(weight.data, 1)
                else:
                    xavier_normal_(
                        weight.data, calculate_gain('leaky_relu', 0.2))
        return self

    def forward(self, x):
        batch, channels, time = x.shape
        x = self.main(x)
        x = self.to_frames(x)
        x = (x ** 2) * torch.sign(x)
        x = x.view(batch, channels, time)
        return x
