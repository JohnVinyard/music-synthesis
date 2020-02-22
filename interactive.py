import numpy as np
import torch
import zounds
import librosa
from featuresynth.data import DataStore
from featuresynth.feature.spectrogram import FilterBankSpectrogram
from featuresynth.audio import MelScalePhaseRecover, GeometricScalePhaseRecover

ds = DataStore('timit', '/hdd/TIMIT', pattern='*.WAV', max_workers=2)
batch_stream = ds.batch_stream(1, {
    'audio': (16384, 1),
    'spectrogram': (64, 256)
})

feature_channels = 256
taps = 1024
pooling = (512, 256)
sr = zounds.SR11025()
geom_scale = zounds.GeometricScale(20, sr.nyquist - 20, 0.05, feature_channels)
mel_scale = zounds.MelScale(
    zounds.FrequencyBand(20, sr.nyquist - 20), feature_channels)

# scaling_factors = np.linspace(0.25, 0.5, feature_channels)
scaling_factors = [0.5] * feature_channels

geom = FilterBankSpectrogram(sr, taps, geom_scale, scaling_factors, pooling)
mel = FilterBankSpectrogram(sr, taps, mel_scale, scaling_factors, pooling)


def compare():
    samples, features = next(batch_stream)
    orig = zounds.AudioSamples(samples.squeeze(), sr)
    f = np.abs(zounds.spectral.stft(orig))
    n_fft, hop_length = pooling
    lm = librosa.feature.melspectrogram(
        orig,
        int(sr),
        n_fft=1024,
        hop_length=hop_length,
        n_mels=feature_channels)
    samples = torch.from_numpy(samples).float()
    g = geom.forward(samples).data.cpu().numpy().squeeze().T
    m = mel.forward(samples).data.cpu().numpy().squeeze().T
    return orig, np.log(g), np.log(m), np.log(f), np.log(lm.T)


def check_recon():
    samples, features = next(batch_stream)
    orig = zounds.AudioSamples(samples.squeeze(), sr)
    samples = torch.from_numpy(orig)
    m = mel.reconstruct(samples)
    g = geom.reconstruct(samples)
    m = zounds.AudioSamples(m.data.cpu().numpy().squeeze(), sr)
    g = zounds.AudioSamples(g.data.cpu().numpy().squeeze(), sr)
    m /= np.abs(m).max()
    g /= np.abs(g).max()
    return orig, m, g


def check():
    samples, features = next(batch_stream)
    orig = zounds.AudioSamples(samples.squeeze(), sr)
    m = MelScalePhaseRecover.from_audio(orig, sr)
    g = GeometricScalePhaseRecover.from_audio(orig, sr)
    return orig, m, g


if __name__ == '__main__':
    # app = zounds.ZoundsApp(globals=globals(), locals=locals())
    # app.start_in_thread(9999)
    # input('Waiting...')
    from featuresynth.experiment import Report
    from featuresynth.experiment.winners import MultiScaleMelGanExperiment
    experiment = MultiScaleMelGanExperiment()
    r = Report(experiment, 'test-generator-report')
    r.generate(ds, 3, sr, regenerate=True)
