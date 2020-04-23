from featuresynth.data import batch_stream
from featuresynth.feature import spectrogram
import zounds
from sklearn.cluster import MiniBatchKMeans
from featuresynth.experiment import FilterBankMultiscaleExperiment
import torch
from featuresynth.audio.transform import fft_frequency_recompose
import numpy as np

samplerate = zounds.SR22050()

def stream(batch_size=64):
    path = '/hdd/musicnet/train_data'
    pattern = '*.wav'

    samplerate = zounds.SR22050()
    feature_spec = {
        'spectrogram': (256, 128)
    }

    feature_funcs = {
        'spectrogram': (spectrogram, (samplerate,))
    }

    bs = batch_stream(
        path, pattern, batch_size, feature_spec, 'spectrogram', feature_funcs)
    return bs


def train_kmeans(km):
    for spec, in stream():
        batch, channels, time = spec.shape
        spec = spec.transpose((0, 2, 1)).reshape((batch * time, channels))
        norms = np.linalg.norm(spec, axis=-1, keepdims=True)
        spec /= norms + 1e-12
        km.partial_fit(spec)
        yield km



if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(8888)

    kmeans = MiniBatchKMeans(n_clusters=256)
    generator = FilterBankMultiscaleExperiment.make_generator()
    generator = FilterBankMultiscaleExperiment \
        .load_generator_weights(generator)

    def check_recon():
        spec, = next(stream(batch_size=1))
        batch, channels, time = spec.shape
        spec = spec.transpose((0, 2, 1)).reshape((batch * time, channels))

        norms = np.linalg.norm(spec, axis=-1, keepdims=True)
        spec /= norms + 1e-12

        indices = kmeans.predict(spec)
        centers = kmeans.cluster_centers_[indices]

        centers *= norms

        recon = centers.reshape((batch, time, channels)).transpose((0, 2, 1))

        bands = generator.forward(torch.from_numpy(recon))
        audio = fft_frequency_recompose(bands, 256 * 256).data.cpu().numpy()
        audio = zounds.AudioSamples(audio.squeeze(), samplerate)
        return spec, recon.squeeze().T, audio


    for trained in train_kmeans(kmeans):
        pass


