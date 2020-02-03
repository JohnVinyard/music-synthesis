import torch
import zounds
from torch.nn import functional as F
import os
from torch import nn
from featuresynth.data import TrainingData
from featuresynth.generator import DDSPGenerator
from featuresynth.util import device
from featuresynth.feature import \
    sr, total_samples, frequency_recomposition, feature_channels, band_sizes, \
    filter_banks, bandpass_filters, slices
import numpy as np
from torch.optim import Adam
from featuresynth.discriminator import Discriminator
import os
from random import choice


def perceptual(x):
    x = F.pad(x, (0, 256))
    x = torch.stft(x, 512, 256, normalized=True)
    # x[:, :, :, 0] = torch.log2(1 + x[:, :, :, 0])
    # print(x)
    return x.contiguous()


def log_magnitude(x):
    return torch.log(torch.abs(x[:, :, :, 1]) + 1e-12)


if __name__ == '__main__':
    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(8888)

    feature_size = 64

    g = DDSPGenerator(feature_size, feature_channels, 128, None, None, None,
                      None) \
        .to(device) \
        .initialize_weights()
    g_optim = Adam(g.parameters(), lr=0.0001, betas=(0, 0.9))

    base_path = '/hdd/musicnet/train_data'
    files = os.listdir(base_path)
    file = choice(files)
    samples = zounds.AudioSamples.from_file(
        os.path.join(base_path, file))[:zounds.Seconds(10)]
    samples = zounds.soundfile.resample(samples, zounds.SR11025())
    start = np.random.randint(0, len(samples) - 16384)
    chunk = samples[start: start + 16384]
    orig = chunk.pad_with_silence()

    target = torch.from_numpy(chunk).to(device).view(1, -1)
    spec_target = perceptual(target)
    # target = F.pad(target, (0, 256))
    # spec_target = torch.stft(target, 512, 256, normalized=True)

    current = None

    while True:
        inp = spec_target.view(64, 257, 2)[:, :256, :1]  # (64, 256, 1)
        inp = inp.permute(2, 1, 0).contiguous()
        output = g(inp).view(1, -1)

        current = zounds.AudioSamples(output.data.cpu().numpy().squeeze(),
                                      zounds.SR11025()).pad_with_silence()
        # current /= (current.max() + 1e-12)

        # output = F.pad(output, (0, 256))
        # spec_output = torch.stft(output, 512, 256, normalized=True)
        spec_output = perceptual(output)

        # loss = -F.cosine_similarity(
        #     spec_output.contiguous().view(1, -1),
        #     spec_target.contiguous().view(1, -1),
        #     dim=-1)
        loss = torch.abs(spec_output - spec_target).sum()

        loss.backward()
        g_optim.step()
        print(loss.item())
