from torch import nn
from torch.nn import functional as F
import torch
from ..util.modules import DilatedStack

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




class PredictiveGenerator(nn.Module):
    def __init__(self):
        super().__init__()


        # self.main = nn.Sequential(
        #     nn.ReflectionPad2d((1, 1, 1, 1)),
        #     nn.Conv2d(1, 8, (2, 2), (1, 1), dilation=(1, 1)),
        #     nn.LeakyReLU(0.2),
        #
        #     nn.ReflectionPad2d((1, 1, 1, 1)),
        #     nn.Conv2d(8, 16, (2, 2), (1, 1), dilation=(2, 2)),
        #     nn.LeakyReLU(0.2),
        #
        #     nn.ReflectionPad2d((2, 2, 2, 2)),
        #     nn.Conv2d(16, 32, (2, 2), (1, 1), dilation=(4, 4)),
        #     nn.LeakyReLU(0.2),
        #
        #     nn.ReflectionPad2d((4, 4, 4, 4)),
        #     nn.Conv2d(32, 64, (2, 2), (1, 1), dilation=(8, 8)),
        #     nn.LeakyReLU(0.2),
        #
        #     nn.ReflectionPad2d((8, 8, 8, 8)),
        #     nn.Conv2d(64, 128, (2, 2), (1, 1), dilation=(16, 16)),
        #     nn.LeakyReLU(0.2),
        #
        #     nn.ReflectionPad2d((16, 16, 16, 16)),
        #     nn.Conv2d(128, 128, (2, 2), (1, 1), dilation=(32, 32)),
        #     nn.LeakyReLU(0.2),
        #
        #     nn.ReflectionPad2d((32, 32, 32, 32)),
        #     nn.Conv2d(128, 1, (2, 2), (1, 1), dilation=(64, 64)),
        # )

        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 16, (3, 3), (2, 2), (1, 1)), # 64
        #     nn.Conv2d(16, 32, (3, 3), (2, 2), (1, 1)), # 32
        #     nn.Conv2d(32, 64, (3, 3), (2, 2), (1, 1)), # 16
        #     nn.Conv2d(64, 128, (3, 3), (2, 2), (1, 1)), # 8
        #     nn.Conv2d(128, 256, (3, 3), (2, 2), (1, 1)), # 4
        #     nn.Conv2d(256, 512, (3, 3), (2, 2), (1, 1)), # 2
        #     nn.Conv2d(512, 1024, (2, 2), (1, 1), (0, 0)), # 1
        # )
        #
        # self.decoder = nn.Sequential(
        #     nn.Conv2d(1024, 512 * 4, (1, 1), (1, 1), (0, 0)), # reshape 2
        #     nn.ConvTranspose2d(512, 256, (4, 4), (2, 2), (1, 1)), # 4
        #     nn.ConvTranspose2d(256, 128, (4, 4), (2, 2), (1, 1)), # 8
        #     nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), (1, 1)), # 16
        #     nn.ConvTranspose2d(64, 32, (4, 4), (2, 2), (1, 1)), # 32
        #     nn.ConvTranspose2d(32, 16, (4, 4), (2, 2), (1, 1)), # 64
        #     nn.ConvTranspose2d(16, 1, (4, 4), (2, 2), (1, 1)), # 128
        # )

        self.encoder = nn.Sequential(
            nn.Conv1d(128, 256, 7, 2, 3), # 64
            nn.Conv1d(256, 512, 7, 2, 3), # 32
            nn.Conv1d(512, 1024, 7, 2, 3), # 16
            nn.Conv1d(1024, 1024, 3, 2, 1), # 8
            nn.Conv1d(1024, 1024, 3, 2, 1), # 4
            nn.Conv1d(1024, 2048, 3, 2, 1), # 2

            nn.ConvTranspose1d(2048, 1024, 4, 2, 1), # 4
            nn.ConvTranspose1d(1024, 1024, 4, 2, 1), # 8
            nn.ConvTranspose1d(1024, 1024, 4, 2, 1), # 16
            nn.ConvTranspose1d(1024, 512, 8, 2, 3), # 32
            nn.ConvTranspose1d(512, 256, 8, 2, 3), # 64
            nn.ConvTranspose1d(256, 128, 8, 2, 3), # 128
        )


        # self.stack = DilatedStack(
        #     128,
        #     512,
        #     2,
        #     [1, 2, 4, 8, 16, 32, 64],
        #     lambda x: F.leaky_relu(x, 0.2),
        #     residual=True)
        # self.to_frames = nn.Conv1d(512, 128, 1, 1, 0)

    # def generate(self, primer, steps=30):
    #     with torch.no_grad():
    #         x = primer[:, :, 128:]
    #
    #         for i in range(steps):
    #             # conditioning is the last 128 frames of the sequence
    #             conditioning = x[:, :, -128:]
    #             predicted = self.inference(conditioning)
    #             x = torch.cat([x, predicted], dim=-1)
    #         return x

    def inference(self, x):
        batch, channels, frames = x.shape
        orig = x = x[:, None, :, :]
        x = self.main(x)
        x = x[:, :, :128, :128]

        # x = torch.cat([orig[..., -1:], x], dim=-1)
        # x = torch.cumsum(x, dim=-1)[..., 1:]

        x = x.view(batch, channels, frames)
        return x

    # def main(self, x):
    #     for i, layer in enumerate(self.encoder):
    #         if i == len(self.encoder) - 1:
    #             x = layer(x)
    #         else:
    #                 x = F.leaky_relu(layer(x), 0.2)
    #
    #     for i, layer in enumerate(self.decoder):
    #         if i == len(self.decoder) - 1:
    #             x = layer(x)
    #         else:
    #             x = F.leaky_relu(layer(x), 0.2)
    #         if i == 0:
    #             x = x.view(-1, 512, 2, 2)
    #     return x

    # def main(self, x):
    #     batch, _, channels, time = x.shape
    #     x = x.view(batch, channels, time)
    #     x = self.stack(x)
    #     x = self.to_frames(x)
    #     x = x.view(batch, 1, channels, time)
    #     return x


    def main(self, x):
        batch, _, channels, time = x.shape
        x = x.view(batch, channels, time)
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if x.shape[1] != 2048 and i < len(self.encoder) - 1:
                x = F.leaky_relu(x, 0.2)
        x = x.view(batch, 1, channels, time)
        return x

    def forward(self, x):
        batch, channels, frames = x.shape
        x = x[:, None, :, :]

        # the conditioning is the first half
        orig = x = x[..., :128]
        x = self.main(x)
        x = x[:, :, :128, :128]


        # START DIFF PRODUCTION
        # x = torch.cat([orig[..., -1:], x], dim=-1)
        # x = torch.cumsum(x, dim=-1)[..., 1:]
        # END DIFF PRODUCTION

        x = torch.cat([orig, x], dim=-1)
        x = x.view(batch, channels, -1)
        return x