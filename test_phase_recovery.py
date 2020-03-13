from featuresynth.audio.representation import BasePhaseRecovery
from featuresynth.data import batch_stream
from featuresynth.feature import audio
import zounds
from librosa.filters import mel
import numpy as np


samplerate = zounds.SR22050()
N_MELS = 128

class MelPhaseRecovery(BasePhaseRecovery):
    basis = mel(
        sr=int(samplerate),
        n_fft=BasePhaseRecovery.N_FFT,
        n_mels=N_MELS)

if __name__ == '__main__':

    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    path = '/hdd/musicnet/train_data'
    pattern = '*.wav'

    samplerate = zounds.SR22050()
    total_samples = 2 ** 17
    total_frames = total_samples // MelPhaseRecovery.HOP
    feature_spec = {
        'audio': (2 ** 17, 1)
    }

    feature_funcs = {
        'audio': (audio, (samplerate,))
    }

    batch_size = 4
    bs = batch_stream(
        path, pattern, batch_size, feature_spec, 'audio', feature_funcs)

    # ensure that I can roundtrip some audio
    batch, = next(bs)
    rep = MelPhaseRecovery.from_audio(batch, samplerate)
    recon = rep.to_audio().squeeze()[0]
    recon = zounds.AudioSamples(recon, samplerate)

    # ensure that I can start from synthesized features
    features = np.zeros((batch_size, N_MELS, total_frames))
    synth_repr = MelPhaseRecovery(features, samplerate)
    synth = synth_repr.to_audio().squeeze()[0]
    synth = zounds.AudioSamples(synth, samplerate)
    input('Waiting...')

