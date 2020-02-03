import torch
import numpy as np


def oscillator_bank(frequency, amplitude, sample_rate):
    """
    frequency and amplitude are (batch, n_oscillators, n_samples)
    sample rate is a scalar
    """
    frequency = torch.clamp(frequency, 20., sample_rate / 2.)
    omegas = frequency * (2 * np.pi)
    omegas = omegas / sample_rate
    phases = torch.cumsum(omegas, dim=-1)
    wavs = torch.sin(phases)
    audio = wavs * amplitude
    audio = torch.sum(audio, dim=1)
    return audio


def noise_bank(shape):
    batch, coeffs, n_samples = shape.shape

    # coeffs contains magnitude and phase information
    window_size = coeffs - 2

    noise = torch.FloatTensor(batch, n_samples, window_size) \
        .normal_(0, 1).to(shape.device)
    noise_coeffs = torch.rfft(noise, 1, normalized=True)
    shape = shape.permute(0, 2, 1).contiguous().view(batch, n_samples, -1, 2)
    # (batch, n_samples, coeffs)
    coeffs = noise_coeffs * shape
    signal = torch.irfft(
        coeffs, 1, normalized=True, signal_sizes=(window_size,))
    signal = signal.view(batch, 1, -1)
    return signal

# def noise_bank(amplitude, response_curves):
#
#     # amplitude is (batch, n_bands, n_samples)
#     # response_curves is (n_bands, n_samples)
#     batch, n_bands, n_samples = amplitude.shape
#
#     # TODO: This can be pre-computed
#     noise = torch.FloatTensor(1, 1, n_samples) \
#         .normal_(0, 1).to(amplitude.device)
#     noise_spectral = torch.rfft(noise, 1, normalized=True)
#     # noise is (1, 1, n_samples)
#     filtered_noise = torch.irfft(
#         response_curves[None, ...] * noise_spectral,
#         1,
#         normalized=True,
#         signal_sizes=(n_samples,))
#     # END: pre-computed
#
#     # filtered_noise is (1, n_bands, n_samples)
#     bands = filtered_noise * amplitude
#     bands = bands.sum(dim=1)
#     return bands
#
