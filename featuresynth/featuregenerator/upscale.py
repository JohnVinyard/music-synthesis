from torch import nn
from torch.nn import functional as F


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
