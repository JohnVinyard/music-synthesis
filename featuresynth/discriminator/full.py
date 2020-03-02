from torch import nn
from torch.nn import functional as F
from torch.nn.init import calculate_gain, xavier_normal_
from torch.nn.utils import weight_norm

def weight_norm(x):
    return x


class FullDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            weight_norm(nn.Conv1d(1, 16, 15, 1, padding=7)),
            weight_norm(nn.Conv1d(16, 64, 41, 4, padding=20, groups=4)),
            weight_norm(nn.Conv1d(64, 256, 41, 4, padding=20, groups=16)),
            weight_norm(nn.Conv1d(256, 1024, 41, 4, padding=20, groups=64)),
            weight_norm(nn.Conv1d(1024, 1024, 41, 4, padding=20, groups=256)),
            weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2))
        )

        self.judge = weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def initialize_weights(self):
        for name, weight in self.named_parameters():
            if weight.data.dim() > 2:
                if 'judge' in name:
                    xavier_normal_(weight.data, 1)
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
