from torch import nn
from torch.nn.init import xavier_normal_, calculate_gain
from torch.nn import functional as F
import torch
from .full import FullDiscriminator


class MelGanDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = FullDiscriminator()
        self.scales = 2

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
        batch = x.shape[0]
        features = []
        judgements = []

        f, j = self.disc(x)
        features.extend(f)
        judgements.append(j)

        for _ in range(self.scales):
            x = F.avg_pool1d(x, 4, 2, 2)
            f, j = self.disc(x)
            features.extend(f)
            judgements.append(j)

        x = torch.cat([j.view(batch, -1) for j in judgements], dim=-1)
        return features, x
