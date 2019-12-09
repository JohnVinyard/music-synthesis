import torch
import numpy as np
from ..util import device
from torch.nn import functional as F
import zounds
import scipy
from scipy.fftpack import dct, idct
from scipy.signal import firwin

total_samples = 16384
band_sizes = [1024, 2048, 4096, 8192, total_samples]

feature_channels = 256

sr = zounds.SR11025()
band = zounds.FrequencyBand(20, sr.nyquist)
scale = zounds.GeometricScale(20, sr.nyquist - 20, 0.05, feature_channels)

chroma_scale = zounds.ChromaScale(band)
chroma_basis = chroma_scale._basis(scale, zounds.HanningWindowingFunc()).T
taps = 1024

fb = zounds.learn.FilterBank(
    sr,
    taps,
    scale,
    np.linspace(0.25, 0.5, len(scale)),
    normalize_filters=False,
    a_weighting=False).to(device)
fb.filter_bank = fb.filter_bank / 10


def generate_filter_banks(band_sizes):
    band_sizes = sorted(band_sizes)
    total_samples = band_sizes[-1]
    n_bands = [128] * 5

    n_taps = 256
    current_low_freq = 20

    for i, size in enumerate(band_sizes):
        ratio = (total_samples / size)
        new_sr = zounds.SampleRate(
            sr.frequency * ratio, sr.duration * ratio)

        if size == total_samples:
            freq_band = zounds.FrequencyBand(current_low_freq,
                                             new_sr.nyquist - 20)
        else:
            freq_band = zounds.FrequencyBand(current_low_freq, new_sr.nyquist)

        bandpass = firwin(
            n_taps,
            [int(new_sr) // 4, (int(new_sr) // 2) - 1],
            fs=int(new_sr),
            pass_zero=False).astype(np.float32)
        bandpass = torch.from_numpy(bandpass).to(device).view(1, 1, n_taps)

        scale = zounds.GeometricScale(
            freq_band.start_hz, freq_band.stop_hz, 0.05, n_bands[i])
        bank = zounds.learn.FilterBank(
            new_sr,
            n_taps,
            scale,
            # values close to zero get good frequency resolution.  Values close
            # to one get good time resolution
            0.25,
            normalize_filters=False,
            a_weighting=False).to(device)

        current_low_freq = freq_band.stop_hz
        yield bank, bandpass


filter_banks, bandpass_filters = \
    list(zip(*list(generate_filter_banks(band_sizes))))

slices = []
for filter_bank in filter_banks:
    band = zounds.FrequencyBand(
        filter_bank.scale.start_hz, filter_bank.scale.stop_hz)
    subset = scale.get_slice(band)
    slices.append(subset)
print(slices)


def transform(samples):
    with torch.no_grad():
        s = torch.from_numpy(samples.astype(np.float32)).to(device)
        result = fb.convolve(s)
        result = F.relu(result)
        result = result.data.cpu().numpy()[..., :samples.shape[-1]]
    return result


def pooled(result):
    padding = np.zeros((result.shape[0], result.shape[1], 256))
    result = np.concatenate([result, padding], axis=-1)
    result = zounds.sliding_window(
        result,
        (result.shape[0], result.shape[1], 512),
        (result.shape[0], result.shape[1], 256))
    result = result.max(axis=-1).transpose((1, 2, 0))
    return result


def mfcc(result):
    result = scipy.fftpack.dct(result, axis=1, norm='ortho')
    return result[:, 1:13, :]


def chroma(result):
    # result will be (batch, channels, time)
    batch, channels, time = result.shape
    result = result.transpose((0, 2, 1)).reshape(-1, channels)
    result = np.dot(result, chroma_basis)
    result = result.reshape((batch, time, -1))
    result = result.transpose((0, 2, 1))
    return result


def low_dim(result, downsample_factor=8):
    # result will be (batch, channels, time)
    arr = np.asarray(result)
    arr = arr.reshape(
        (result.shape[0], -1, downsample_factor, result.shape[-1]))
    s = arr.mean(axis=1)
    return s


def compute_features(samples):
    spectral = transform(samples)
    return pooled(spectral).astype(np.float32)


def frequency_decomposition(samples, sizes):
    sizes = sorted(sizes)
    batch_size = samples.shape[0]
    samples = samples.reshape((-1, samples.shape[-1]))
    coeffs = dct(samples, axis=-1, norm='ortho')
    positions = [0] + sizes
    slices = [
        slice(positions[i], positions[i + 1])
        for i in range(len(positions) - 1)]
    bands = []
    for size, sl in zip(sizes, slices):
        new_coeffs = np.zeros((batch_size, size), dtype=np.float32)
        new_coeffs[:, sl] = coeffs[:, sl]
        resampled = idct(new_coeffs, axis=-1, norm='ortho')
        bands.append(resampled)

    return bands


def frequency_recomposition(bands, total_size):
    batch_size = bands[0].shape[0]
    bands = sorted(bands, key=lambda band: len(band))
    final = np.zeros((batch_size, total_size))
    for i, band in enumerate(bands):
        coeffs = dct(band, axis=-1, norm='ortho')
        new_coeffs = np.zeros((batch_size, total_size))
        new_coeffs[:, :band.shape[-1]] = coeffs
        ups = idct(new_coeffs, axis=-1, norm='ortho')
        final += ups
    return final
