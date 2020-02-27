from torch import nn
from torch.nn.init import xavier_normal_, calculate_gain
from torch.nn import functional as F
from .full import FullDiscriminator


class MelGanDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = FullDiscriminator()
        self.scales = 2

    def forward(self, x):
        features = []
        judgements = []

        f, j = self.disc(x)
        features.append(f)
        judgements.append(j)

        for _ in range(self.scales):
            x = F.avg_pool1d(x, kernel_size=4, stride=2, padding=2)
            f, j = self.disc(x)
            features.append(f)
            judgements.append(j)

        return features, judgements
