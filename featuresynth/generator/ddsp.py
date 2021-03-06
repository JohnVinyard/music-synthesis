import torch
import numpy as np
from torch.nn import functional as F


def oscillator_bank(frequency, amplitude, sample_rate):
    """
    frequency and amplitude are (batch, n_oscillators, n_samples)
    sample rate is a scalar
    """
    # constrain frequencies
    frequency = torch.clamp(frequency, 20., sample_rate / 2.)

    # translate frequencies in hz to radians
    omegas = frequency * (2 * np.pi)
    omegas = omegas / sample_rate


    phases = torch.cumsum(omegas, dim=-1)
    wavs = torch.sin(phases)

    audio = wavs * amplitude
    audio = torch.sum(audio, dim=1)
    return audio





def noise_bank2(x):
    # TODO: Understand and apply stuff about periodic, zero-phase, causal
    # TODO: Understand and apply stuff about windowing the filter coefficients
    # windows
    batch, magnitudes, samples = x.shape
    window_size = (magnitudes - 1) * 2
    hop_size = window_size // 2
    total_samples = hop_size * samples

    # (batch, frames, coeffs, 2)

    # create the noise
    noise = torch.FloatTensor(batch, total_samples).uniform_(-1, 1).to(x.device)
    # window the noise
    noise = F.pad(noise, (0, hop_size))
    noise = noise.unfold(-1, window_size, hop_size)
    noise_coeffs = torch.rfft(noise, 1, normalized=True)
    # (batch frames, coeffs, 2)

    x = x.permute(0, 2, 1)[..., None]
    # apply the filter in the frequency domain
    filtered = noise_coeffs * x

    # recover the filtered noise in the time domain
    audio = torch.irfft(
        filtered, 1, normalized=True, signal_sizes=(window_size,))
    audio = overlap_add(audio[:, None, :, :], apply_window=True)
    audio = audio[..., :total_samples]
    audio = audio.view(batch, 1, -1)
    return audio


from scipy.signal import hann


def np_overlap_add(x, apply_window=True, hop_size=None):
    batch, channels, frames, samples = x.shape

    if apply_window:
        # window = np.hamming(samples)
        # window = hann(samples, False)
        window = np.hamming(samples)
        x = x * window[None, None, None, :]

    hop_size = hop_size or samples // 2
    first_half = x[:, :, :, :hop_size].reshape(batch, channels, -1)
    second_half = x[:, :, :, hop_size:].reshape(batch, channels, -1)
    first_half = np.pad(
        first_half, ((0, 0), (0, 0), (0, hop_size)), mode='constant')
    second_half = np.pad(
        second_half, ((0, 0), (0, 0), (hop_size, 0)), mode='constant')
    output = first_half + second_half
    return output


def overlap_add(x, apply_window=True):
    batch, channels, frames, samples = x.shape

    if apply_window:
        window = torch.from_numpy(hann(samples, False)).to(x.device).float()
        # window = torch.hamming_window(samples, periodic=False).to(x.device)
        # window = torch.hann_window(samples, periodic=False).to(x.device)
        x = x * window[None, None, None, :]

    hop_size = samples // 2
    first_half = x[:, :, :, :hop_size].contiguous().view(batch, channels, -1)
    second_half = x[:, :, :, hop_size:].contiguous().view(batch, channels, -1)
    first_half = F.pad(first_half, (0, hop_size))
    second_half = F.pad(second_half, (hop_size, 0))
    output = first_half + second_half
    return output





def smooth_upsample2(x, size):
    batch, channels, frames = x.shape
    hop_size = size // frames
    window_size = hop_size * 2

    window = torch.hamming_window(window_size, periodic=True).to(x.device)

    amps = x.view(batch, channels * frames)
    scaled_windows = amps[..., None] * window[None, None, :]
    scaled_windows = scaled_windows.view(batch, channels, frames, window_size)
    output = overlap_add(scaled_windows, apply_window=False)
    output = output[:, :, :-hop_size]
    return output

