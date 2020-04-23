from torch import nn
from torch.nn import functional as F
import torch
from ..util.modules import DilatedStack, Reshape


class SimpleGenerator(nn.Module):
    def __init__(self):
        super().__init__()



        self.main = nn.Sequential(
            # summarize
            Reshape((1, 128, 128)),
            nn.Conv2d(1, 32, (3, 9), (2, 2), (1, 4)),  # (64, 64)
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, (3, 9), (2, 2), (1, 4)),  # (32, 32)
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, (3, 9), (2, 2), (1, 4)),  # (16, 16)
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, (3, 9), (2, 2), (1, 4)),  # (8, 8)
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, (3, 9), (2, 2), (1, 4)),  # (4, 4)

            # transform
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2),


            # generate
            nn.ConvTranspose2d(512, 256, (4, 8), (2, 2), (1, 3)), # (8, 8)
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, (4, 8), (2, 2), (1, 3)),  # (16, 16)
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, (4, 8), (2, 2), (1, 3)),  # (32, 32)
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, (4, 8), (2, 2), (1, 3)),  # (64, 64)
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, (4, 8), (2, 2), (1, 3)),  # (128, 128)
        )


    # def generate(self, primer, steps=30):
    #     with torch.no_grad():
    #         x = primer[:, :, 128:]
    #         for i in range(steps):
    #             # conditioning is the last 128 frames of the sequence
    #             conditioning = x[:, :, -128:]
    #             predicted = self.inference(conditioning)
    #             x = torch.cat([x, predicted], dim=-1)
    #         return x
    #
    # def inference(self, x):
    #     batch, channels, frames = x.shape
    #     x = self.main(x)
    #     x = x.view(batch, channels, frames)
    #     return x

    def forward(self, x):
        batch, channels, time = x.shape
        orig = x = x[:, :, :128]
        x = self.main(x)
        x = x.view(batch, channels, 128)
        x = torch.cat([orig, x], dim=-1)
        return x



class ResidualBlock(nn.Module):
    def __init__(self, features, dilation, kernel_size=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.features = features
        padding = max(1, dilation // 2)
        self.conv = nn.Conv1d(
            features,
            features,
            2,
            stride=1,
            padding=padding,
            dilation=dilation,
            padding_mode='reflection')
        self.gate = nn.Conv1d(
            features,
            features,
            2,
            stride=1,
            padding=padding,
            dilation=dilation,
            padding_mode='reflection')
        self.final = nn.Conv1d(features, features, 1, 1, 0)

    def forward(self, x):
        orig = x
        x = F.tanh(self.conv(x)) * F.sigmoid(self.gate(x))
        x = x[..., :orig.shape[-1]]
        x = self.final(x)
        return orig + x


class ResidualStack(nn.Module):
    def __init__(self, features, dilations, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.features = features
        self.main = nn.Sequential(
            *[ResidualBlock(features, d) for d in dilations]
        )

    def forward(self, x):
        features = []
        for layer in self.main:
            x = layer(x)
            features.append(x)
        x = sum(features)
        x = F.leaky_relu(x, 0.2)
        return x


class PredictiveFrameGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        # self.encoder = nn.Sequential(
        #     Reshape((128, 128)),
        #     nn.Conv1d(128, 256, 7, 2, 3), # 64
        #     nn.LeakyReLU(0.2),
        #     nn.Conv1d(256, 512, 7, 2, 3), # 32
        #     nn.LeakyReLU(0.2),
        #     nn.Conv1d(512, 1024, 7, 2, 3), # 16
        #     nn.LeakyReLU(0.2),
        #     nn.Conv1d(1024, 1024, 3, 2, 1), # 8
        #     nn.LeakyReLU(0.2),
        #     nn.Conv1d(1024, 1024, 3, 2, 1), # 4
        #     nn.LeakyReLU(0.2),
        #     nn.Conv1d(1024, 2048, 3, 2, 1), # 2
        #
        #     nn.ConvTranspose1d(2048, 1024, 4, 2, 1), # 4
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose1d(1024, 1024, 4, 2, 1), # 8
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose1d(1024, 1024, 4, 2, 1), # 16
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose1d(1024, 512, 8, 2, 3), # 32
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose1d(512, 512, 8, 2, 3), # 64
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose1d(512, 512, 8, 2, 3), # 128
        # )
        #
        # self.frame_decoder = nn.Sequential(
        #     Reshape((128, 4, 128)),
        #     nn.ConvTranspose2d(128, 128, (4, 1), (2, 1), (1, 0)), # (8, 128)
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose2d(128, 64, (4, 1), (2, 1), (1, 0)),  # (16, 128)
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose2d(64, 32, (4, 1), (2, 1), (1, 0)),  # (32, 128)
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose2d(32, 16, (4, 1), (2, 1), (1, 0)),  # (64, 128)
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose2d(16, 1, (4, 1), (2, 1), (1, 0)),  # (128, 128)
        # )


        self.embedding = nn.Conv1d(128, 512, 1, 1, 0)
        self.stack = ResidualStack(
            512,
            [1, 2, 4, 8, 16, 32, 64, 1, 2, 4, 8, 16, 32, 64, 1],
            2)

        self.final = nn.Sequential(
            nn.Conv1d(512, 512, 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 128, 1, 1, 0)
        )

    def generate(self, primer, steps=30):
        with torch.no_grad():
            x = primer[:, :, 128:]
            for i in range(steps):
                # conditioning is the last 128 frames of the sequence
                conditioning = x[:, :, -128:]
                predicted = self.inference(conditioning)
                x = torch.cat([x, predicted], dim=-1)
            return x

    def inference(self, x):
        batch, channels, frames = x.shape
        x = self.main(x)
        x = x.view(batch, channels, frames)
        return x


    def main(self, x):
        x = self.embedding(x)
        x = self.stack(x)
        x = self.final(x)
        return x


    def forward(self, x):
        batch, channels, frames = x.shape

        # the conditioning is the first half
        orig = x = x[..., :128]
        x = self.main(x)
        x = x[:, :128, :128]

        x = torch.cat([orig, x], dim=-1)
        x = x.view(batch, channels, -1)
        return x


class ARPredictiveGenerator(nn.Module):
    def __init__(self, frames, out_channels, noise_dim, channels):
        super().__init__()
        self.out_channels = out_channels
        self.noise_dim = noise_dim
        self.channels = channels
        self.frames = frames

        # self.frame_embedding = nn.Sequential(
        #     Reshape((1, 128, 128)),
        #     nn.Conv2d(1, 16, (3, 1), (2, 1), (1, 0)),  # (64, 128)
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(16, 32, (3, 1), (2, 1), (1, 0)),  # (32, 128)
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(32, 64, (3, 1), (2, 1), (1, 0)),  # (16, 128)
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(64, 128, (3, 1), (2, 1), (1, 0)),  # (8, 128)
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(128, 256, (3, 1), (2, 1), (1, 0)),  # (4, 128)
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(256, 256, (4, 1), (1, 1), (0, 0)),  # (1, 128)
        #     nn.LeakyReLU(0.2),
        #     Reshape((256, 128))
        # )

        self.frame_embedding = nn.Conv1d(128, 32, 1, 1, 0)

        self.main = nn.Sequential(
            nn.Conv1d(channels, channels, 2, dilation=1, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=2, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=4, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=8, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=16, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=32, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=64, bias=False),
        )

        self.gate = nn.Sequential(
            nn.Conv1d(channels, channels, 2, dilation=1, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=2, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=4, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=8, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=16, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=32, bias=False),
            nn.Conv1d(channels, channels, 2, dilation=64, bias=False),
        )

        # self.final = nn.Sequential(
        #     nn.Conv1d(channels, channels, 1, 1, 0, bias=False),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv1d(channels, channels, 1, 1, 0, bias=False),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv1d(channels, channels * 4, 1, 1, 0, bias=False),
        #     nn.LeakyReLU(0.2),
        #     Reshape((channels, 4)),
        #     nn.ConvTranspose1d(channels, 1024, 4, 2, 1, bias=False), # 8
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose1d(1024, 512, 4, 2, 1, bias=False), # 16
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose1d(512, 256, 4, 2, 1, bias=False), # 32
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose1d(256, 256, 4, 2, 1, bias=False), # 64
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose1d(256, 128, 4, 2, 1, bias=False), # 128
        #     nn.LeakyReLU(0.2),
        #     nn.Conv1d(128, 128, 3, 1, 1, bias=False)
        # )

        self.final = nn.Sequential(

            # transform embedding
            nn.Conv1d(channels, channels, 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels * 4, 1, 1, 0),
            nn.LeakyReLU(0.2),
            Reshape((channels, 4)),

            nn.ConvTranspose1d(channels, 1024, 4, 2, 1), # 8
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(1024, 512, 4, 2, 1),  # 16
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(512, 256, 4, 2, 1),  # 32
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(256, 128, 4, 2, 1),  # 64
            nn.LeakyReLU(0.2),
            # embedding bottleneck
            nn.ConvTranspose1d(128, 32, 4, 2, 1),  # 128
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 128, 1, 1, 0)


            # scale up in the time dimension
            # nn.ConvTranspose1d(channels, 1024, 4, 2, 1), # 8
            # nn.LeakyReLU(0.2),
            # nn.ConvTranspose1d(1024, 512, 4, 2, 1),  # 16
            # nn.LeakyReLU(0.2),
            # nn.ConvTranspose1d(512, 256, 4, 2, 1),  # 32
            # nn.LeakyReLU(0.2),
            # nn.ConvTranspose1d(256, 128, 4, 2, 1),  # 64
            # nn.LeakyReLU(0.2),
            # nn.ConvTranspose1d(128, 64 * 4, 4, 2, 1),  # (batch, 64, 128)
            # nn.LeakyReLU(0.2),
            # Reshape((64, 4, 128)),
            #
            # # scale up in the frequency dimension
            # nn.ConvTranspose2d(64, 32, (4, 1), (2, 1), (1, 0)), # 8
            # nn.LeakyReLU(0.2),
            # nn.ConvTranspose2d(32, 32, (4, 1), (2, 1), (1, 0)),  # 16
            # nn.LeakyReLU(0.2),
            # nn.ConvTranspose2d(32, 32, (4, 1), (2, 1), (1, 0)),  # 32
            # nn.LeakyReLU(0.2),
            # nn.ConvTranspose2d(32, 16, (4, 1), (2, 1), (1, 0)),  # 64
            # nn.LeakyReLU(0.2),
            # nn.ConvTranspose2d(16, 1, (4, 1), (2, 1), (1, 0)),  # 64
            # nn.LeakyReLU(0.2),
            #
            # Reshape((128, 128))
        )

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
        x = self.frame_embedding(x)
        features = []
        for layer, gate in zip(self.main, self.gate):
            p = layer.dilation[0]
            padded = F.pad(x, (p, 0))
            z = F.tanh(layer(padded)) * F.sigmoid(gate(padded))
            features.append(z)
            if x.shape[1] == z.shape[1]:
                x = z + x
            else:
                x = z
        x = sum(features)[:, :, -1:]
        x = self.final(x)
        return x

    def forward(self, x):
        batch, channels, frames = x.shape
        orig = x = x.view(batch, self.out_channels, -1)[:, :, :128]

        # features = []
        # for layer, gate in zip(self.main, self.gate):
        #     p = layer.dilation[0]
        #     padded = F.pad(x, (p, 0))
        #     z = F.tanh(layer(padded)) * F.sigmoid(gate(padded))
        #     features.append(z)
        #     if x.shape[1] == z.shape[1]:
        #         x = z + x
        #     else:
        #         x = z
        #
        # x = sum(features)[:, :, -1:]
        # x = self.final(x)

        x = self.inference(x)

        x = torch.cat([orig, x], dim=-1)
        return x