from torch import nn
from torch.nn.init import xavier_normal_, calculate_gain
import torch


class MDCTGenerator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, 512, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 256, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 256, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )

        self.to_frames = nn.Conv1d(256, 256, 3, 1, 1)

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
        x = self.main(x)
        x = self.to_frames(x)
        x = (x ** 2) * torch.sign(x)
        return x
