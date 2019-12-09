from featuresynth.data import TrainingData
from featuresynth.feature import \
    frequency_decomposition, frequency_recomposition, band_sizes, sr, \
    total_samples, filter_banks
import numpy as np
import torch
from featuresynth.util import device
import zounds


# def test_frequency_decomposition(total_samples, band_sizes, sr):
#     synth = zounds.SineSynthesizer(sr)
#     samples = synth.synthesize(
#         sr.frequency * total_samples, [55, 110, 220, 440, 880, 1660, 1660 * 2])
#     batch = np.repeat(samples[None, :], 8, axis=0)
#     bands = frequency_decomposition(batch, band_sizes)
#     recomposed = frequency_recomposition(bands, total_samples)
#     recomposed = zounds.AudioSamples(recomposed[0], sr).pad_with_silence()
#     bands = [band[0] for band in bands]
#     return bands, recomposed


def test_synthetic(batch_size):
    synth = zounds.SineSynthesizer(sr)
    samples = synth.synthesize(
        sr.frequency * total_samples, [55, 110, 220, 440, 880, 1660, 1660 * 2])
    batch = np.repeat(samples[None, :], batch_size, axis=0)
    bands = frequency_decomposition(batch, band_sizes)
    recomposed = frequency_recomposition(bands, total_samples)
    recomposed = zounds.AudioSamples(recomposed[0], sr).pad_with_silence()
    return samples, recomposed


def test_filter_bank_recon(samples, return_spectral=False):
    samples = samples[:1, ...]

    samples /= samples.max()

    bands = frequency_decomposition(samples, band_sizes)
    new_bands = []
    spectral = []

    for band, fb in zip(bands, filter_banks):
        band = torch.from_numpy(band).float().to(device)
        sp = fb.convolve(band)
        spectral.append(sp.data.cpu().numpy())
        band = fb.transposed_convolve(sp)
        new_bands.append(band.data.cpu().numpy())

    final = frequency_recomposition(new_bands, total_samples)
    orig = zounds.AudioSamples(samples.squeeze(), sr)
    final = zounds.AudioSamples(final.squeeze(), sr)
    final /= final.max()
    if return_spectral:
        return orig, final, spectral
    else:
        return orig, final


if __name__ == '__main__':
    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    batch_size = 4
    td = TrainingData(
        '/hdd/musicnet/train_data',
        batch_size=batch_size,
        total_samples=total_samples,
        sr=sr)
    app.start_in_thread()

    synth_orig, synth_recon = test_synthetic(batch_size)
    _, _, synth_spec = test_filter_bank_recon(synth_orig[None, :], return_spectral=True)

    while True:
        bands, _ = next(td.batch_stream())
        bands = [b.data.cpu().numpy() for b in bands]
        samples = frequency_recomposition(bands, total_samples)
        orig, recon, spectral = \
            test_filter_bank_recon(samples, return_spectral=True)
        input('Waiting...')
