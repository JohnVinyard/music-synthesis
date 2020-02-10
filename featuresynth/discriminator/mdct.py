from torch import nn
from torch.nn.init import xavier_normal_, calculate_gain
from torch.nn import functional as F


class MDCTDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, 512, 3, 1, 1),
            nn.Conv1d(512, 256, 3, 1, 1),
            nn.Conv1d(256, 128, 3, 1, 1),
        )
        self.judge = nn.Conv1d(128, 1, 3, 1, 1)

    def initialize_weights(self):
        for name, weight in self.named_parameters():
            if weight.data.dim() > 2:
                if 'judge' in name:
                    xavier_normal_(weight.data, calculate_gain('tanh'))
                else:
                    xavier_normal_(
                        weight.data, calculate_gain('leaky_relu', 0.2))
        return self

    def forward(self, x):
        features = []
        for layer in self.main:
            x = F.leaky_relu(layer(x), 0.2)
            features.append(x)
        x = self.judge(x)
        return features, x

