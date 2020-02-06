from torch import nn
from torch.nn.utils import weight_norm
from torch.nn import functional as F
import numpy as np
import torch
from torch.nn.init import xavier_normal_, calculate_gain, orthogonal_
import zounds
import math

from ..util import device
from ..util.modules import UpsamplingStack, LearnedUpSample, UpSample


def weight_norm(x):
    return x


class GeneratorBlock(nn.Module):
    """
    Apply a series of increasingly dilated convolutions with optinal upscaling
    at the end
    """

    def __init__(self, dilations, channels, kernel_size, upsample_factor=2):
        super().__init__()
        self.upsample_factor = upsample_factor
        self.kernel_size = kernel_size
        self.channels = channels
        self.dilations = dilations
        layers = []
        for i, d in enumerate(self.dilations):
            padding = (kernel_size * d) // 2
            c = nn.Conv1d(
                channels,
                channels,
                kernel_size,
                dilation=d,
                padding=padding,
                bias=False)
            layers.append(c)

        # Batch norm seems to be responsible for all the annoying high-frequency
        # chirps or blips
        self.main = nn.Sequential(*[weight_norm(layer) for layer in layers])
        # No overlap helps to avoid buzzing/checkerboard artifacts

        # BE SURE TO SWITCH OUT INITIALIZATION TOO!
        self.activation = lambda x: F.leaky_relu(x, 0.2)

        self.upsampler = weight_norm(nn.ConvTranspose1d(
            channels,
            channels,
            self.upsample_factor * 2,
            stride=self.upsample_factor,
            padding=self.upsample_factor // 2,
            bias=False))

    def forward(self, x):
        batch_size = x.shape[0]
        dim = x.shape[-1]
        x = x.view(x.shape[0], self.channels, -1)

        for i, layer in enumerate(self.main):
            t = layer(x)
            # This residual seems to be very important, at least when using a
            # mostly-positive activation like ReLU
            x = self.activation(x + t[..., :dim])

        if self.upsample_factor > 1:
            x = self.upsampler(x)
            x = self.activation(x)

        return x


class SineChannelGenerator(nn.Module):
    def __init__(self, input_size, target_size, in_channels, channels, fb, sl,
                 bandpass):
        super().__init__()
        self.bandpass = bandpass
        self.sl = sl
        self.channels = channels
        self.in_channels = in_channels
        self.target_size = target_size
        self.input_size = input_size

        filter_bank_channels = fb.filter_bank.shape[0]

        scale = fb.scale
        # start_hz = scale.frequency_band.start_hz
        # stop_hz = scale.frequency_band.stop_hz
        # hz_range = stop_hz - start_hz
        # sample_range = self.target_size // 2  # nyquist frequency
        # self.band_starts = np.array([int((b.start_hz / hz_range) * sample_range) for b in scale]).astype(np.float32)
        # self.band_ranges = np.array([int((b.bandwidth / hz_range) * sample_range) for b in scale]).astype(np.float32)

        band_starts = np.geomspace(
            self.target_size // 4, self.target_size // 2, len(scale) + 1)
        bandwidths = np.diff(band_starts)

        self.band_starts = band_starts.astype(np.float32)[:-1]
        self.band_ranges = bandwidths.astype(np.float32)

        print(self.target_size, '==================================')
        print(self.band_starts)
        print(self.band_ranges)

        self.embedding_layer = \
            weight_norm(
                nn.Conv1d(sl.stop - sl.start, channels, 1, 1, bias=False))

        self.main = nn.Sequential(
            nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
            nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
            nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
        )

        self.amps = nn.Conv1d(channels, filter_bank_channels, 1, 1, bias=False)
        self.frequencies = nn.Conv1d(channels, filter_bank_channels, 1, 1,
                                     bias=False)
        self.phases = nn.Conv1d(channels, filter_bank_channels, 1, 1,
                                bias=False)

        # self.amps = nn.Sequential(
        #     nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
        #     nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
        #     nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
        #     nn.Conv1d(channels, filter_bank_channels, 1, 1, bias=False)
        # )
        #
        # self.frequencies = nn.Sequential(
        #     nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
        #     nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
        #     nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
        #     nn.Conv1d(channels, filter_bank_channels, 1, 1, bias=False)
        # )
        #
        # self.phases = nn.Sequential(
        #     nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
        #     nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
        #     nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
        #     nn.Conv1d(channels, filter_bank_channels, 1, 1, bias=False)
        # )

        self.scale = nn.Parameter(torch.FloatTensor(1).fill_(0.0001))

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1, self.input_size)

        # frequency slice
        x = x[:, self.sl, :]

        embedded = F.leaky_relu(self.embedding_layer(x), 0.2)

        for layer in self.main:
            embedded = F.leaky_relu(layer(embedded), 0.2)

        amps = F.sigmoid(self.amps(embedded))
        freqs = F.sigmoid(self.frequencies(embedded))
        phases = F.sigmoid(self.phases(embedded))

        # amps = embedded
        # freqs = embedded
        # phases = embedded
        #
        # for i, layer in enumerate(self.amps):
        #     if i == len(self.amps) - 1:
        #         amps = F.sigmoid(layer(amps))
        #     else:
        #         amps = F.leaky_relu(layer(amps), 0.2)
        #
        # for i, layer in enumerate(self.frequencies):
        #     if i == len(self.frequencies) - 1:
        #         freqs = F.sigmoid(layer(freqs))
        #     else:
        #         freqs = F.leaky_relu(layer(freqs), 0.2)
        #
        # for i, layer in enumerate(self.phases):
        #     if i == len(self.phases) - 1:
        #         phases = F.sigmoid(layer(phases))
        #     else:
        #         phases = F.leaky_relu(layer(phases), 0.2)

        phases = phases * 2 * math.pi
        # phases = torch.cat(
        #     [torch.zeros(batch_size, phases.shape[1], 1).to(x.device), phases], dim=-1)
        phases = torch.cumsum(phases, dim=-1)

        # freqs is (batch, filterbank_channels, feature_size)

        band_starts = torch.from_numpy(self.band_starts).to(x.device)
        band_ranges = torch.from_numpy(self.band_ranges).to(x.device)
        freqs = band_starts[None, :, None] + (
        freqs * band_ranges[None, :, None])

        amps = amps * torch.abs(self.scale)

        amps = F.upsample(amps, size=self.target_size, mode='nearest')
        freqs = F.upsample(freqs, size=self.target_size, mode='nearest')
        phases = F.upsample(phases, size=self.target_size, mode='nearest')

        # TODO: Compute starting points and *ranges* for each band in the
        # filter bank in terms of cycles per sample
        c = torch.linspace(-math.pi, math.pi, self.target_size).to(x.device)

        # (batch, filterbank_channels, target_size)
        # (target_size)
        x = (freqs * c[None, None, :]) + phases
        result = amps * torch.sin(x)
        result = result.sum(dim=1, keepdim=True)
        return result


class DilatedChannelGenerator(nn.Module):
    def __init__(self, input_size, target_size, in_channels, channels, fb, sl,
                 bandpass):
        super().__init__()
        self.bandpass = bandpass
        self.sl = sl
        self.channels = channels
        self.in_channels = in_channels
        self.target_size = target_size
        self.input_size = input_size

        self.embedding_layer = \
            weight_norm(
                nn.Conv1d(sl.stop - sl.start, channels, 1, 1, bias=False))

        self.scale_factor = self.target_size // self.input_size
        #
        # self.upsampler = nn.ConvTranspose1d(
        #     channels,
        #     channels,
        #     self.scale_factor * 2,
        #     self.scale_factor,
        #     self.scale_factor // 2,
        #     bias=False)

        self.feature = nn.Sequential(
            GeneratorBlock([1, 3, 9], channels, 3, upsample_factor=1),
            GeneratorBlock([1, 1, 1], channels, 3, upsample_factor=1),
        )

        self.main = nn.Sequential(
            GeneratorBlock([1, 3, 9], channels, 3, upsample_factor=1),
            GeneratorBlock([1, 3, 9], channels, 3, upsample_factor=1),
            GeneratorBlock([1, 1], channels, 3, upsample_factor=1),
        )

        filter_bank_channels = fb.filter_bank.shape[0]

        self.to_samples = weight_norm(nn.Conv1d(
            channels, filter_bank_channels, 1, stride=1, padding=1, bias=False))
        # KLUDGE: There must be a better way to exclude model parameters?
        self.fb = [fb]
        self.scale = nn.Parameter(torch.FloatTensor(1).fill_(0.0001))

        self.activation = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1, self.input_size)

        # frequency slice
        x = x[:, self.sl, :]

        embedded = self.activation(self.embedding_layer(x))
        for layer in self.feature:
            embedded = layer(embedded)

        embedded = F.upsample(embedded, scale_factor=self.scale_factor,
                              mode='nearest')

        for i, layer in enumerate(self.main):
            embedded = layer(embedded)

        samples = self.to_samples(embedded)
        samples = self.fb[0].transposed_convolve(samples)
        samples = samples * torch.abs(self.scale)

        # bandpass filter the results to prevent out-of-band signal
        # samples = F.conv1d(
        #     samples, self.bandpass, padding=self.bandpass.shape[-1] // 2)

        return samples[..., :self.target_size]


class ChannelGenerator(nn.Module):
    def __init__(self, input_size, target_size, in_channels, channels, fb, sl,
                 bandpass):
        super().__init__()
        self.bandpass = bandpass
        self.sl = sl
        self.channels = channels
        self.in_channels = in_channels
        self.target_size = target_size
        self.input_size = input_size

        self.embedding_layer = \
            weight_norm(
                nn.Conv1d(sl.stop - sl.start, channels, 1, 1, bias=False))

        n_layers = int(np.log2(target_size) - np.log2(input_size))
        layers = []

        # layer_dict = {
        #     1024: [2, 2, 2, 2, 1],
        #     2048: [4, 2, 2, 2, 1],
        #     4096: [4, 4, 2, 2, 1],
        #     8192: [8, 4, 2, 2, 1],
        #     16384: [8, 8, 2, 2, 1]
        #
        # }

        for _ in range(n_layers):
            block = GeneratorBlock([1, 3, 9], channels, 3, upsample_factor=2)
            layers.append(block)
        layers.append(GeneratorBlock([1, 3, 9], channels, 3, upsample_factor=1))

        # for upsample_factor in layer_dict[self.target_size]:
        #     block = GeneratorBlock(
        #         [1, 3, 9], channels, 3, upsample_factor=upsample_factor)
        #     layers.append(block)
        # layers.append(nn.Conv1d(channels, channels, 7, 1, 3, bias=False))

        self.main = nn.Sequential(*layers)

        filter_bank_channels = fb.filter_bank.shape[0]

        self.to_samples = weight_norm(nn.Conv1d(
            channels, filter_bank_channels, 1, stride=1, padding=1, bias=False))
        # KLUDGE: There must be a better way to exclude model parameters?
        self.fb = [fb]
        self.scale = nn.Parameter(torch.FloatTensor(1).fill_(0.0001))

        self.activation = lambda x: F.leaky_relu(x, 0.2)

    def test(self):
        batch_size = 8
        inp = np.random.normal(
            0, 1, (batch_size, self.in_channels, self.input_size))
        t = torch.from_numpy(inp.astype(np.float32))
        out = self.forward(t).data.cpu().numpy()
        print(out.shape)
        assert (batch_size, 1, self.target_size) == out.shape

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1, self.input_size)

        # frequency slice
        x = x[:, self.sl, :]

        embedded = self.activation(self.embedding_layer(x))

        for i, layer in enumerate(self.main):
            embedded = layer(embedded)

        samples = self.to_samples(embedded)
        samples = self.fb[0].transposed_convolve(samples)
        samples = samples * torch.abs(self.scale)

        # bandpass filter the results to prevent out-of-band signal
        # samples = F.conv1d(
        #     samples, self.bandpass, padding=self.bandpass.shape[-1] // 2)

        return samples[..., :self.target_size]





class Generator(nn.Module):
    def __init__(
            self,
            input_size,
            in_channels,
            channels,
            output_sizes,
            filter_banks,
            slices,
            bandpass_filters):

        super().__init__()
        self.bandpass_filters = bandpass_filters
        self.slices = slices
        self.channels = channels
        self.output_sizes = output_sizes
        self.in_channels = in_channels
        self.input_size = input_size

        generators = []

        for size, fb, sl, bpf in zip(output_sizes, filter_banks, slices,
                                     bandpass_filters):
            generators.append(
                DilatedChannelGenerator(input_size, size, in_channels, channels,
                                        fb,
                                        sl, bpf))
        self.generators = nn.Sequential(*generators)

    def initialize_weights(self):
        for name, weight in self.named_parameters():
            if weight.data.dim() > 2:
                if 'samples' in name:
                    xavier_normal_(weight.data, 1)
                else:
                    xavier_normal_(
                        weight.data, calculate_gain('leaky_relu', 0.2))

    def test(self):
        batch_size = 8
        inp = torch.FloatTensor(
            *(batch_size, self.in_channels, self.input_size))
        out = self.forward(inp)
        assert len(self.output_sizes) == len(out)
        for item, size in zip(out, self.output_sizes):
            print(item.shape)
            assert (batch_size, 1, size) == item.shape

    def forward(self, x):
        x = x.view(-1, self.in_channels, self.input_size)
        return \
            {size: g(x) for size, g in zip(self.output_sizes, self.generators)}
        # bands = [g(x) for g in self.generators]
        # return bands


class SimpleGenerator(nn.Module):
    """
    This generator is only suitable for evaluating discriminators
    """

    def __init__(
            self,
            input_size,
            in_channels,
            channels,
            output_sizes,
            filter_banks):

        super().__init__()
        self.output_sizes = output_sizes
        self.filter_banks = filter_banks

        generators = []
        for size, fb in zip(self.output_sizes, filter_banks):
            filter_bank_channels = fb.filter_bank.shape[0]
            param = nn.Parameter(
                torch.FloatTensor(1, filter_bank_channels, size + 1))
            generators.append(nn.ParameterList(parameters=[param]))
            self.register_parameter(f'p{size}', param)
        self.generators = nn.Sequential(*generators)

    def initialize_weights(self):
        for name, weight in self.named_parameters():
            weight.data.normal_(0, 0.02)

    def forward(self, x):
        # result = []
        result = {}
        for size, fb in zip(self.output_sizes, self.filter_banks):
            p = getattr(self, f'p{size}')
            samples = fb.transposed_convolve(p)
            result[size] = samples
            # result.append(samples)
        # result = [getattr(self, f'p{size}') for size in self.output_sizes]
        return result
