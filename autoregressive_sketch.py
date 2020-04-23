from featuresynth.data import batch_stream
from featuresynth.audio import MDCT
from featuresynth.audio.representation import BasePhaseRecovery
from featuresynth.feature import audio
from librosa.filters import mel
import zounds
from sklearn.cluster import MiniBatchKMeans
from itertools import islice
import numpy as np
import math

"""
What we'll try:
- MDCT and mel spectrogram representations
- K-means using no regularization, std and fractal representation
- PCA on each repr then K-Means
"""

N_MELS = 128
samplerate = zounds.SR22050()

FRACTAL_WINDOW_SIZE = 2

class MelPhaseRecovery(BasePhaseRecovery):
    basis = mel(
        sr=int(samplerate),
        n_fft=BasePhaseRecovery.N_FFT,
        n_mels=N_MELS)




def repr_stream(repr_class):
    path = '/hdd/musicnet/train_data'
    pattern = '*.wav'

    total_samples = 2 ** 17
    audio_channels = 1
    feature_spec = {
        'audio': (total_samples, audio_channels)
    }
    feature_funcs = {
        'audio': (audio, (samplerate,))
    }
    batch_size = 2
    bs = batch_stream(
        path, pattern, batch_size, feature_spec, 'audio', feature_funcs)

    for samples,  in bs:
        rep = repr_class.from_audio(samples, samplerate)
        yield rep


def fractal(x, window_size):
    examples, channels = x.shape
    l = math.log(channels, window_size)
    if l % 1 != 0:
        raise ValueError(f'window size must be a logarithm '
                         f'of {channels} but was {window_size}')
    output = []
    while x.shape[-1] > 1:
        x = zounds.sliding_window(
            x, (1, window_size), (1, window_size), flatten=False).squeeze(axis=2)
        # (examples, n_windows, window)
        norms = np.linalg.norm(x, axis=-1, keepdims=True)
        # print(x.shape, norms.shape)
        # (examples, n_windows, 1)
        x = x / (norms + 1e-12)
        output.append(x.reshape((examples, -1)))
        x = norms.reshape((examples, -1))

    output.append(x)
    return output[::-1]

def packed_fractal(x, window_size):
    output = fractal(x, window_size)
    return np.concatenate(output, axis=-1)

def fractal_recon(output, window_size):
    examples = output[0].shape[0]
    norms = output[0]
    windows = output[1]

    norms = windows * norms

    for i in range(2, len(output)):
        windows = output[i]
        norms = norms.reshape((examples, -1, 1))
        windows = windows.reshape((examples, -1, window_size))
        norms = windows * norms

    norms = norms.reshape((examples, -1))
    return norms

def unpacked_fractal_recon(x, window_size):
    output = []
    start = 0
    i = 0
    while True:
        size = window_size ** i
        slce = x[:, start:start + size]
        if slce.shape[-1] == 0:
            break
        output.append(slce)
        start += size
        i += 1
    recon = fractal_recon(output, window_size)
    return recon

def unit_norm(x):
    norms = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    return norms, x / norms


def packed_channels(x):
    """
    (batch, channels, time) => (examples, channels)
    """
    _, channels, _ = x.shape
    x = x.transpose((0, 2, 1)).reshape((-1, channels))
    return x

def unpacked_channels(x, time_dim):
    """
    (examples, channels) => (batch, channels, time)
    """
    _, channels = x.shape
    x = x.reshape((-1, time_dim, channels)).transpose((0, 2, 1))
    return x



def learn_clusters(stream, n_clusters=512, n_iterations=10000):
    kmeans_batch_size = n_clusters * 2
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, batch_size=kmeans_batch_size)

    for i, rep in enumerate(islice(stream, n_iterations)):
        data = rep.data # (batch, channels, time)
        data = packed_channels(data)
        data = packed_fractal(data, FRACTAL_WINDOW_SIZE)
        kmeans.partial_fit(data[:, 1:])
        yield i, kmeans

def do_recon(km, time_dim):
    rep = next(make_stream())
    data = rep.data
    data = packed_channels(data)
    data = packed_fractal(data, FRACTAL_WINDOW_SIZE)
    norms = data[:, :1]

    indices = km.predict(data[:, 1:])

    centroids = km.cluster_centers_[indices]
    centroids = np.concatenate([norms, centroids], axis=1)
    centroids = unpacked_fractal_recon(centroids, FRACTAL_WINDOW_SIZE)



    centroids = unpacked_channels(centroids, time_dim)
    recon_rep = rep.__class__(centroids, samplerate)
    return rep, recon_rep


def make_stream():
    return repr_stream(MelPhaseRecovery)

if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(8888)

    rs = make_stream()

    for i, km in learn_clusters(rs):
        print(f'kmeans iter {i}')


    input('Waiting...')