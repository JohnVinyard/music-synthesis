from featuresynth.audio.representation import BasePhaseRecovery
from featuresynth.data import iter_files
from featuresynth.feature import audio
import zounds
from librosa.filters import mel
from random import choice

samplerate = zounds.SR22050()

class MelPhaseRecovery(BasePhaseRecovery):
    basis = mel(
        sr=int(samplerate),
        n_fft=1024,
        n_mels=256)

if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    path = '/hdd/musicnet/train_data'
    pattern = '*.wav'
    file_path = choice(list(iter_files(path, pattern)))
    n_samples = 2 ** 17
    raw_samples = audio(file_path, samplerate)[n_samples:n_samples*2]
    samples = zounds.AudioSamples(raw_samples, samplerate)

    rep = MelPhaseRecovery.from_audio(samples, samples.samplerate)
    recon = zounds.AudioSamples(rep.to_audio().squeeze(), samplerate)
    input('Waiting...')

