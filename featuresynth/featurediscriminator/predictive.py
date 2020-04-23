from torch import nn
from torch.nn import functional as F

from ..util.modules import DilatedStack, Reshape

class SimpleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # embedding
            Reshape((1, 128, 256)),
            # look at narrow frequency bands for long time-span
            nn.Conv2d(1, 32, (3, 9), (2, 4), (1, 3)), # (64, 64)
            nn.LeakyReLU(0.2),

            # look at wide frequency bands
            nn.Conv2d(32, 64, (3, 9), (2, 2), (1, 4)), # (32, 32)
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, (3, 9), (2, 2), (1, 4)),  # (16, 16)
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, (3, 9), (2, 2), (1, 4)),  # (8, 8)
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, (3, 9), (2, 2), (1, 4)),  # (4, 4)
        )



    def forward(self, x, conditioning):
        batch, channels, time = x.shape # (b, 128, 256)
        x = self.main(x)
        features = []
        return [], x

class FrameSpectrogramFeatureDiscriminator(nn.Module):
    def __init__(self, feature_channels):
        super().__init__()
        self.feature_channels = feature_channels

        # collapse frames into a sequence of frame embeddings
        # self.main = nn.Sequential(
        #     Reshape((1, 128, 256)),
        #     nn.Conv2d(1, 16, (3, 1), (2, 1), (1, 0)), # (64, 256)
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(16, 32, (3, 1), (2, 1), (1, 0)),  # (32, 256)
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(32, 64, (3, 1), (2, 1), (1, 0)),  # (16, 256)
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(64, 128, (3, 1), (2, 1), (1, 0)),  # (8, 256)
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(128, 256, (3, 1), (2, 1), (1, 0)),  # (4, 256)
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(256, 256, (4, 1), (1, 1), (0, 0)),  # (4, 256)
        #     nn.LeakyReLU(0.2),
        #     Reshape((256, 256))
        # )

        # linear frame embedding
        self.main = nn.Conv1d(128, 32, 1, 1, 0)

        # look at sequences of embeddings
        # TODO: Should this scale up to 81?
        self.stack = DilatedStack(
            32,
            32,
            3,
            [1, 3, 9, 27, 81, 1, 1],
            activation=lambda x: F.leaky_relu(x, 0.2),
            residual=True)
        self.judge = nn.Conv1d(32, 1, 1, 1, 0)

    def forward(self, x, conditioning):
        # NOTE: Conditioning is meaningless here, as the generator is
        # unconditioned
        x = self.main(x)
        features, x = self.stack(x, return_features=True)
        x = self.judge(x)
        return features, x