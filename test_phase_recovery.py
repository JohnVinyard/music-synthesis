from featuresynth.audio.representation import BasePhaseRecovery, RawAudio
from featuresynth.data import batch_stream
from featuresynth.feature import audio
import zounds
from librosa.filters import mel
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans


class SpectrogramCompressor(object):
    def __init__(self, n_components, batch_size):
        super().__init__()
        self._meanstd = StandardScaler()
        self._pca = IncrementalPCA(n_components, batch_size=batch_size)
        self._kmeans = MiniBatchKMeans(n_clusters=512, batch_size=batch_size)

    def partial_fit(self, data):
        self._meanstd.partial_fit(data)
        data = self._meanstd.transform(data)
        self._pca.partial_fit(data)
        data = self._pca.transform(data)
        self._kmeans.partial_fit(data)


    def transform(self, data):
        data = self._meanstd.transform(data)
        data = self._pca.transform(data)
        data = self._kmeans.predict(data)
        return data

    def inverse_transform(self, data):
        data = self._kmeans.cluster_centers_[data]
        data = self._pca.inverse_transform(data)
        data = self._meanstd.inverse_transform(data)
        return data



samplerate = zounds.SR22050()
N_MELS = 128

class MelPhaseRecovery(BasePhaseRecovery):
    basis = mel(
        sr=int(samplerate),
        n_fft=BasePhaseRecovery.N_FFT,
        n_mels=N_MELS)


class IdentityPhaseReovery(BasePhaseRecovery):
    basis = None


def stream(total_samples=8192):
    path = '/hdd/musicnet/train_data'
    pattern = '*.wav'

    samplerate = zounds.SR22050()
    # total_samples = 8192
    feature_spec = {
        'audio': (total_samples, 1)
    }

    feature_funcs = {
        'audio': (audio, (samplerate,))
    }

    batch_size = 32
    bs = batch_stream(
        path, pattern, batch_size, feature_spec, 'audio', feature_funcs)
    for batch,  in bs:
        transformed = IdentityPhaseReovery.from_audio(batch, samplerate)
        yield batch, transformed


def train_pca(pca):
    for i, packed in enumerate(stream()):
        samples, rep = packed
        data = rep.data.reshape((-1, rep.data.shape[-1]))
        pca.partial_fit(data)
        print(f'iter {i}')
        yield pca

if __name__ == '__main__':

    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    # pca = IncrementalPCA(n_components=N_MELS, batch_size=1024)
    pca = SpectrogramCompressor(N_MELS, batch_size=1024)

    def check_recon():
        samples, item = next(stream(total_samples=2**17))
        batch, time, channels = item.data.shape
        flattened = item.data.reshape((-1, channels))
        reduction = pca.transform(flattened)
        recon = pca.inverse_transform(reduction)
        recon = recon.reshape((batch, time, channels))
        rp = IdentityPhaseReovery(recon, samplerate)
        return \
            reduction.reshape((batch, time, -1)), \
            rp, \
            MelPhaseRecovery.from_audio(samples, samplerate)

    for trained in train_pca(pca):
        pass

    # path = '/hdd/musicnet/train_data'
    # pattern = '*.wav'
    #
    # samplerate = zounds.SR22050()
    # total_samples = 2 ** 17
    # total_frames = total_samples // MelPhaseRecovery.HOP
    # feature_spec = {
    #     'audio': (2 ** 17, 1)
    # }
    #
    # feature_funcs = {
    #     'audio': (audio, (samplerate,))
    # }
    #
    # batch_size = 4
    # bs = batch_stream(
    #     path, pattern, batch_size, feature_spec, 'audio', feature_funcs)
    #
    # # ensure that I can roundtrip some audio
    # batch, = next(bs)
    # orig = RawAudio(batch, samplerate)
    #
    # rep = MelPhaseRecovery.from_audio(batch, samplerate)
    # recon = rep.to_audio().squeeze()[0]
    # recon = zounds.AudioSamples(recon, samplerate)
    #
    # # ensure that I can start from synthesized features
    # features = np.zeros((batch_size, N_MELS, total_frames))
    # synth_repr = MelPhaseRecovery(features, samplerate)
    # synth = synth_repr.to_audio().squeeze()[0]
    # synth = zounds.AudioSamples(synth, samplerate)
    # input('Waiting...')
    #
