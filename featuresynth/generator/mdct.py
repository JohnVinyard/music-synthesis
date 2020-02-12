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

        self.to_frames = nn.Conv1d(channels, 256, 7, 1, 3)
        self.to_frames_gate = nn.Conv1d(channels, 256, 7, 1, 3)


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
        x = self.main(x)
        x = F.tanh(self.to_frames(x)) * F.tanh(self.to_frames_gate(x))
        x = x.view(batch, channels, time)
        return x
