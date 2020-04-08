from torch import nn
from torch.nn import functional as F


class OneDimensionalSpectrogramGenerator(nn.Module):
    def __init__(self, out_channels, noise_dim):
        super().__init__()
        self.noise_dim = noise_dim
        self.out_channels = out_channels
        self.initial = nn.Linear(noise_dim, 8 * 1024)
        self.main = nn.Sequential(
            nn.ConvTranspose1d(1024, 512, 4, 2, 1), # 16
            nn.ConvTranspose1d(512, 512, 4, 2, 1), # 32
            nn.ConvTranspose1d(512, 512, 4, 2, 1), # 64
            nn.ConvTranspose1d(512, 256, 4, 2, 1), # 128
            nn.ConvTranspose1d(256, 256, 4, 2, 1), # 256
            nn.ConvTranspose1d(256, 128, 4, 2, 1), # 512
        )

    def forward(self, x):
        x = x.view(-1, self.noise_dim)
        x = F.leaky_relu(self.initial(x), 0.2)
        x = x.view(-1, 1024, 8)
        for i, layer in enumerate(self.main):
            if i == len(self.main) - 1:
                x = layer(x)
            else:
                x = F.leaky_relu(layer(x), 0.2)
        return x


class NearestNeighborOneDimensionalSpectrogramGenerator(nn.Module):
    def __init__(self, out_channels, noise_dim):
        super().__init__()
        self.noise_dim = noise_dim
        self.out_channels = out_channels
        self.initial = nn.Linear(noise_dim, 8 * 1024)
        self.main = nn.Sequential(
            nn.Conv1d(1024, 1024, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2), # 16

            nn.Conv1d(1024, 512, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2), # 32

            nn.Conv1d(512, 512, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2), # 64

            nn.Conv1d(512, 256, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2), # 128

            nn.Conv1d(256, 256, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2), # 256

            nn.Conv1d(256, 256, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2), # 512

            nn.Conv1d(256, 256, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 128, 7, 1, 3),
        )

    def forward(self, x):
        x = x.view(-1, self.noise_dim)
        x = F.leaky_relu(self.initial(x), 0.2)
        x = x.view(-1, 1024, 8)
        x = self.main(x)
        return x


class SpectrogramFeatureGenerator(nn.Module):
    def __init__(self, out_channels, noise_dim):
        super().__init__()
        self.noise_dim = noise_dim
        self.out_channels = out_channels

        self.initial = nn.Linear(noise_dim, 4 * 4 * 1024)
        self.stack = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, (4, 4), (2, 2), (1, 1)),
            # (8, 8)
            nn.ConvTranspose2d(512, 256, (4, 4), (2, 2), (1, 1)),
            # (16, 16)
            nn.ConvTranspose2d(256, 128, (4, 4), (2, 2), (1, 1)),
            # (32, 32)
            nn.ConvTranspose2d(128, 128, (4, 4), (2, 2), (1, 1)),
            # (64, 64)
            nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), (1, 1)),
            # (128, 128)
            nn.ConvTranspose2d(64, 32, (3, 4), (1, 2), (1, 1)),
            # (128, 256)
            nn.ConvTranspose2d(32, 1, (3, 4), (1, 2), (1, 1)),
            # (128, 512)
        )

    def forward(self, x):
        x = x.view(-1, self.noise_dim)
        x = F.leaky_relu(self.initial(x), 0.2)
        x = x.view(x.shape[0], -1, 4, 4)
        for i, layer in enumerate(self.stack):
            if i == len(self.stack) - 1:
                x = layer(x)
            else:
                x = F.leaky_relu(layer(x), 0.2)

        x = x.view(x.shape[0], self.out_channels, -1)
        return x


