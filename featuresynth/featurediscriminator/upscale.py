import torch
from torch import nn
from torch.nn.init import xavier_normal_, calculate_gain
from torch.nn import functional as F
from ..util.modules import \
    DilatedStack, DownsamplingStack, flatten_channels, unflatten_channels
import numpy as np

class EnergyDiscriminator(nn.Module):
    """
    This discriminator judges whether loudness time series are real or fake
    """

    def __init__(self, frames, feature_channels, channels, n_judgements, ae):
        super().__init__()
        self.ae = [ae]
        self.n_judgements = n_judgements
        self.channels = channels
        self.feature_channels = feature_channels
        self.frames = frames

        self.loudness = nn.Sequential(
            nn.Conv1d(1, channels, 3, 1, dilation=1, padding=1, bias=False),
            nn.Conv1d(channels, channels, 3, 1, dilation=3, padding=3,
                      bias=False),
            nn.Conv1d(channels, channels, 3, 1, dilation=9, padding=9,
                      bias=False),
            nn.Conv1d(channels, channels, 3, 1, dilation=27, padding=27,
                      bias=False),
            nn.Conv1d(channels, channels, 3, 1, dilation=1, padding=1,
                      bias=False),
            nn.Conv1d(channels, channels, 3, 1, dilation=1, padding=1,
                      bias=False),
        )

        self.loudness_judge = nn.Conv1d(channels, 1, 1, 1, 0, bias=False)



    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, self.frames)
        for layer in self.loudness:
            x = F.leaky_relu(layer(x), 0.2)
        x = self.loudness_judge(x)
        return x


class DiscriminatorBlock(nn.Module):
    def __init__(
            self,
            condition_channels,
            out_channels,
            channels,
            cond_kernel_size=0,
            output_kernel_size=0):

        super().__init__()
        self.output_kernel_size = output_kernel_size
        self.cond_kernel_size = cond_kernel_size
        self.channels = channels
        self.out_channels = out_channels
        self.condition_channels = condition_channels

        c_channels = condition_channels
        o_channels = out_channels

        if cond_kernel_size > 0:
            self.cond_encoder = DownsamplingStack(
                start_size=condition_channels,
                target_size=1,
                scale_factor=2,
                layer_func=self._build_spec_layer,
                activation=lambda x: F.leaky_relu(x, 0.2))
            c_channels = channels

        if output_kernel_size > 0:
            self.out_encoder = DownsamplingStack(
                start_size=out_channels,
                target_size=1,
                scale_factor=2,
                layer_func=self._build_spec_layer,
                activation=lambda x: F.leaky_relu(x, 0.2))
            o_channels = channels

        self.embedding = nn.Conv1d(
            c_channels + o_channels, channels, 1, 1, 0, bias=False)

        self.stack = DilatedStack(
            channels,
            channels,
            3,
            [1, 3, 9, 27, 1],
            lambda x: F.leaky_relu(x, 0.2),
            residual=True)
        self.judge = nn.Conv1d(channels, 1, 1, 1, 0, bias=False)

    def _build_spec_layer(self, i, curr_size, out_size, first, last):
        kernel_size = 7 if curr_size > 8 else 3
        padding = kernel_size // 2
        return nn.Conv1d(
            in_channels=1 if first else self.channels,
            out_channels=self.channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            bias=False)

    def _run_spec_stack(self, x, stack):
        batch_size = x.shape[0]
        x = flatten_channels(x, channels_as_time=True)
        x = stack(x)
        x = unflatten_channels(x, batch_size)
        return x

    def forward(self, conditioning, output):
        batch_size = output.shape[0]
        features = []
        judgements = []

        # per-frame normalize
        c_mx, _ = conditioning.max(dim=1, keepdim=True)
        o_mx, _ = output.max(dim=1, keepdim=True)
        conditioning = conditioning / (c_mx + 1e-12)
        output = output / (o_mx + 1e-12)

        if self.cond_kernel_size > 0:
            conditioning = self._run_spec_stack(conditioning, self.cond_encoder)

        if self.output_kernel_size > 0:
            output = self._run_spec_stack(output, self.out_encoder)

        if conditioning is not None:
            x = torch.cat([conditioning, output], dim=1)
            x = self.embedding(x)
            f, x = self.stack(x, return_features=True)
            features.extend(f)
            x = self.judge(x)
            judgements.append(x)

        x = torch.cat([j.view(batch_size, -1) for j in judgements], dim=-1)
        return features, x


class Discriminator(nn.Module):
    def __init__(self, frames, feature_channels, channels, n_judgements, ae):
        super().__init__()
        self.ae = ae
        self.n_judgements = n_judgements
        self.channels = channels
        self.feature_channels = feature_channels
        self.frames = frames

        self.loudness = EnergyDiscriminator(
            frames, feature_channels, channels, n_judgements, ae)
        self.four = DiscriminatorBlock(1, 4, channels)
        self.sixteen = DiscriminatorBlock(4, 16, channels)
        self.sixty_four = DiscriminatorBlock(16, 64, channels)
        self.two_fifty_six = DiscriminatorBlock(64, 256, channels)

        self.layers = {
            1: self.loudness,
            4: self.four,
            16: self.sixteen,
            64: self.sixty_four,
            256: self.two_fifty_six
        }

    def initialize_weights(self):
        for name, weight in self.named_parameters():
            if weight.data.dim() > 2:
                # weight.data.normal_(0, 0.02)
                if 'judge' in name:
                    xavier_normal_(weight.data, 1)
                else:
                    xavier_normal_(
                        weight.data, calculate_gain('leaky_relu', 0.2))
        return self

    def forward(self, x):
        pass


class LowResDiscriminator(nn.Module):
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

        # self.stack = nn.Sequential(
        #     nn.Conv2d(1, 64, (5, 5), (2, 2), (2, 2), bias=False),
        #     nn.Conv2d(64, 128, (5, 5), (2, 2), (2, 2), bias=False),
        #     nn.Conv2d(128, 128, (5, 5), (2, 2), (2, 2), bias=False),
        #     nn.Conv2d(128, 128, (5, 5), (2, 2), (2, 2), bias=False),
        #     nn.Conv2d(128, 256, (5, 5), (2, 2), (2, 2), bias=False),
        #     nn.Conv2d(256, 512, (3, 3), (2, 2), (1, 1), bias=False),
        #     nn.Conv2d(512, 1024, (3, 3), (2, 2), (1, 1), bias=False),
        #     nn.Conv2d(1024, 2048, (2, 2), (1, 1), (0, 0), bias=False),
        # )
        #
        # self.judge = nn.Linear(2048, 1, bias=False)

    # def initialize_weights(self):
    #     for name, weight in self.named_parameters():
    #         if weight.data.dim() > 2:
    #             # weight.data.normal_(0, 0.02)
    #             if 'judge' in name:
    #                 xavier_normal_(weight.data, 1)
    #             else:
    #                 xavier_normal_(
    #                     weight.data, calculate_gain('leaky_relu', 0.2))
    #     return self

    def forward(self, x):

        x = self.stack(x)

        # x = x.view(x.shape[0], 1, self.feature_channels, self.frames)
        # for layer in self.stack:
        #     x = F.leaky_relu(layer(x), 0.2)
        # x = x.view(x.shape[0], -1)
        x = self.judge(x)
        return x
