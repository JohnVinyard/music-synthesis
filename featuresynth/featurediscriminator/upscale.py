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
        self.judge = nn.Conv1d(channels, 1, 1, 1, 0)

    def forward(self, x, conditioning):
        # NOTE: Conditioning is meaningless here, as the generator is
        # unconditioned
        features, x = self.stack(x, return_features=True)
        x = self.judge(x)
        return features, x



class CollapseSpectrogramFeatureDiscriminator(nn.Module):
    def __init__(self, feature_channels):
        super().__init__()
        self.feature_channels = feature_channels

        self.main = nn.Sequential(
            nn.Conv1d(feature_channels, 256, 7, 2, 3), # 256
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, 7, 2, 3), # 128
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 512, 7, 2, 3), # 64
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 1024, 7, 2, 3), # 32
            nn.LeakyReLU(0.2),
            nn.Conv1d(1024, 1024, 7, 2, 3), # 16
            nn.LeakyReLU(0.2),
            nn.Conv1d(1024, 1024, 7, 2, 3), # 8
            nn.LeakyReLU(0.2),
            nn.Conv1d(1024, 2048, 3, 2, 1), # 8
            nn.LeakyReLU(0.2),
            nn.Conv1d(2048, 1, 3, 2, 1), # 4
            # nn.LeakyReLU(0.2),
            # nn.Conv1d(2048, 1, 2, 1, 0), # 2
        )



    def forward(self, x, conditioning):
        # NOTE: Conditioning is meaningless here, as the generator is
        # unconditioned
        x = self.main(x)
        features = []
        return features, x



class TwoDimFeatureDiscriminator(nn.Module):
    def __init__(self, feature_channels):
        super().__init__()
        self.feature_channels = feature_channels
        self.main = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), (2, 2), (1, 1)), # 64, 256
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, (3, 3), (2, 2), (1, 1)), # 32, 128
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, (3, 3), (2, 2), (1, 1)), # 16, 64
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, (3, 3), (2, 2), (1, 1)), # 8, 32
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, (3, 3), (2, 2), (1, 1)), # 4, 16
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, (3, 3), (1, 2), (1, 1)), # 4, 8
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 2048, (3, 3), (1, 2), (1, 1)), # 4, 4
            nn.LeakyReLU(0.2),
            nn.Conv2d(2048, 1, (4, 4), (1, 1), (0, 0)), # 4, 4
        )

    def forward(self, x, conditioning):
        batch, channels, time = x.shape
        x = x[:, None, :, :]
        x = self.main(x)
        features = []
        x = x.view(batch, 1, 1)
        return features, x
