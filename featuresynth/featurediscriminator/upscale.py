from torch import nn
from torch.nn import functional as F

from ..util.modules import DilatedStack


class SpectrogramFeatureDiscriminator(nn.Module):
    def __init__(self, feature_channels, channels):
        super().__init__()
        self.channels = channels
        self.feature_channels = feature_channels

        self.stack = DilatedStack(
            feature_channels,
            channels,
            3,
            [1, 3, 9, 27, 1, 1],
            activation=lambda x: F.leaky_relu(x, 0.2),
            residual=True)
        self.judge = nn.Conv1d(channels, 1, 1, 1, 0, bias=False)

    def forward(self, x, conditioning):
        # NOTE: Conditioning is meaningless here, as the generator is
        # unconditioned
        x = self.stack(x)
        x = self.judge(x)
        features = []
        return features, x
