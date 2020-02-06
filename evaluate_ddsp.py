import torch
import zounds
from torch.nn import functional as F
import os
from torch import nn
from featuresynth.data import TrainingData
from featuresynth.generator import DDSPGenerator
from featuresynth.generator.ddsp import np_overlap_add
from featuresynth.util import device
from featuresynth.feature import \
    sr, total_samples, frequency_recomposition, feature_channels, band_sizes, \
    filter_banks, bandpass_filters, slices, compute_features
import numpy as np
from torch.optim import Adam
from featuresynth.discriminator import Discriminator
import os
from random import choice

sr = zounds.SR11025()
scale = zounds.MelScale(zounds.FrequencyBand(20, sr.nyquist), 128)
fb = zounds.learn.FilterBank(
    sr,
    128,
    scale,
    np.linspace(0.25, 0.5, len(scale)),
    normalize_filters=False,
    a_weighting=False).to(device)
fb.filter_bank = fb.filter_bank / 10


def perceptual(x, window=512, log_mag=False):
    x = F.pad(x, (0, window // 2))
    x = torch.stft(x, window, window // 2, normalized=True)
    if log_mag:
        x = torch.log(1e-12 + torch.abs(x[:, :, 0]))
    return x.contiguous().view(x.shape[0], -1)


# def multiscale_loss(x, y, scales=[512, 256, 128, 64]):
#     loss = None
#     for scale in scales:
#         l = torch.abs(perceptual(x, scale) - perceptual(y, scale)).sum()
#         if loss is None:
#             loss = l
#         else:
#             loss += l
#     return loss


def perceptual2(x):
    x = fb.forward(x, normalize=False)
    x = fb.temporal_pooling(x, 512, 256)
    return x


def multiscale_loss(x, y):
    x = fb.forward(x, normalize=False)
    y = fb.forward(y, normalize=False)
    loss = torch.abs(x - y).sum()

    x = fb.temporal_pooling(x, 512, 256)
    y = fb.temporal_pooling(y, 512, 256)
    loss += torch.abs(x - y).sum()

    return loss


def get_filter_coeffs(window_size, n):
    n_coeffs = (window_size // 2) + 1
    start = np.zeros(n_coeffs)
    end = np.zeros(n_coeffs)

    start[0:2] = np.hamming(2)

    end[1:10] = np.hamming(9)

    lines = []
    for i in range(n_coeffs):
        lines.append(np.linspace(start[i], end[i], num=n)[None, :])
    coeffs = np.concatenate(lines, axis=0)[None, :, :]
    return coeffs


def test_spectral_filtering():
    # (1, 129, 64)
    total_samples = 16384
    window_size = 32
    hop_size = 16

    coeffs = get_filter_coeffs(window_size, total_samples // hop_size)

    noise = np.random.uniform(-1, 1, total_samples)
    noise = np.pad(noise, ((0, hop_size),), mode='constant')
    windowed = zounds.sliding_window(noise, window_size, hop_size)
    # (1, 64, 256)
    noise_coeffs = np.fft.rfft(windowed, axis=-1, norm='ortho')
    # (1, 64, 129)

    filtered = coeffs.transpose((0, 2, 1)) * noise_coeffs
    recovered = np.fft.irfft(filtered, axis=-1, norm='ortho')
    samples = np_overlap_add(recovered[:, None, :, :], apply_window=True)
    samples = samples.squeeze()[:total_samples]
    # (1, 64, 256)
    return zounds.AudioSamples(samples, zounds.SR11025()).pad_with_silence()


# def test_spectral_filtering_torch():
#     coeffs = get_filter_coeffs(256)
#     coeffs = torch.from_numpy(coeffs).float()
#
#     noise = torch.FloatTensor(16384).uniform_(-1, 1)
#     windowed = noise.unfold(-1, 256, 256)
#     noise_coeffs = torch.rfft(windowed, 1, normalized=True)
#     noise_coeffs = noise_coeffs.view(1, 64, 129, 2)
#
#     coeffs = coeffs.permute(0, 2, 1)[..., None]
#
#     filtered = coeffs * noise_coeffs
#     recovered = torch.irfft(filtered, 1, normalized=True, signal_sizes=(256,))
#     recovered = recovered.view(-1)
#     return zounds.AudioSamples(
#         recovered.data.cpu().numpy().squeeze(),
#         zounds.SR11025()
#     ).pad_with_silence()


real_noise = zounds.AudioSamples(
    np.random.uniform(-1, 1, 16384),
    zounds.SR11025()
).pad_with_silence()
spec_test = test_spectral_filtering()
# spec_test /= (spec_test.max() + 1e-12)
# torch_spec_test = test_spectral_filtering_torch()
# torch_spec_test /= (torch_spec_test.max() + 1e-12)

if __name__ == '__main__':
    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(8888)

    feature_size = 64

    g = DDSPGenerator(feature_size, feature_channels, 128, None, None, None,
                      None) \
        .to(device) \
        .initialize_weights()
    g_optim = Adam(g.parameters(), lr=0.001, betas=(0, 0.9))

    base_path = '/hdd/musicnet/train_data'
    files = os.listdir(base_path)
    file = choice(files)
    samples = zounds.AudioSamples.from_file(
        os.path.join(base_path, file))[:zounds.Seconds(10)]
    samples = zounds.soundfile.resample(samples, zounds.SR11025())

    start = np.random.randint(0, len(samples) - 16384)
    chunk = samples[start: start + 16384]
    chunk /= (chunk.max() + 1e-12)
    # chunk = spec_test[:16384].astype(np.float32)
    orig = chunk.pad_with_silence()

    target = torch.from_numpy(chunk).to(device).view(1, -1)

    current = None
    inp = compute_features(chunk)
    inp = torch.from_numpy(inp).to(device)
    cond = inp.data.cpu().numpy().squeeze().T

    while True:
        g_optim.zero_grad()
        harmonic, noise, loudness, frequency, filter_coeffs = g(inp)
        output = (harmonic + noise).view(1, -1)

        h = harmonic.data.cpu().numpy().squeeze()
        n = noise.data.cpu().numpy().squeeze()
        h = zounds.AudioSamples(h, zounds.SR11025()).pad_with_silence()
        n = zounds.AudioSamples(n, zounds.SR11025()).pad_with_silence()
        fc = filter_coeffs.data.cpu().numpy().squeeze().T

        l = loudness.data.cpu().numpy().squeeze()
        f = frequency.data.cpu().numpy().squeeze()

        current = zounds.AudioSamples(output.data.cpu().numpy().squeeze(),
                                      zounds.SR11025()).pad_with_silence()

        # loss = multiscale_loss(target, output, scales=[512])
        # loss = torch.abs(perceptual2(target) - perceptual2(output)).sum()
        # loss = torch.abs(
        #     perceptual(target, log_mag=True) -
        #     perceptual(output, log_mag=True)).sum()

        loss = multiscale_loss(target, output)

        loss.backward()

        # for grad in zounds.learn.util.gradients(g):
        #     print(grad)

        g_optim.step()
        print(loss.item())
