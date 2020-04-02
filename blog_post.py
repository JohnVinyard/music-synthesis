from featuresynth.data import batch_stream
from featuresynth.feature import audio
from featuresynth.audio.transform import fft_frequency_decompose, fft_resample
from featuresynth.audio import RawAudio
import zounds
import torch
import numpy as np
from matplotlib import pyplot as plt

path = '/hdd/musicnet/train_data'
pattern = '*.wav'
total_samples = 2 ** 17

samplerate = zounds.SR22050()
feature_spec = {
    'audio': (total_samples, 1)
}

feature_funcs = {
    'audio': (audio, (samplerate,))
}

batch_size = 1
bs = batch_stream(
    path, pattern, batch_size, feature_spec, 'audio', feature_funcs)

if __name__ == '__main__':
    # app = zounds.ZoundsApp(locals=locals(), globals=globals())
    # app.start_in_thread(9999)
    # samples, = next(bs)
    # samples = torch.from_numpy(samples)
    # min_size = 2 ** (np.log2(total_samples) - 4)
    # bands = fft_frequency_decompose(samples, min_size)
    # samples = zounds.AudioSamples(samples.squeeze(), samplerate)
    # input('Waiting...')

    n_bands = 5
    sr = samplerate
    for i in range(n_bands):
        start_hz = 0 if i == (n_bands - 1) else sr.nyquist / 2
        stop_hz = sr.nyquist
        n_samples = int(zounds.Seconds(1) / sr.frequency)
        print(n_samples, start_hz, stop_hz)
        sr *= 2