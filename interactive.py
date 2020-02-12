import zounds
import numpy as np
from featuresynth.data import DataStore

ds = DataStore('timit', '/hdd/TIMIT', pattern='*.WAV', max_workers=2)
batch_stream = ds.batch_stream(
        4,
        {'audio': 16384, 'spectrogram': 64},
        ['audio', 'spectrogram'],
        {'audio': 1, 'spectrogram': 256})

def overlap_add(x):
    win_size = x.shape[-1]
    half_win = win_size // 2
    a = x[..., :half_win]
    b = x[..., half_win:]

    def pad_values(axis, value):
        padding = [(0, 0) for _ in range(x.ndim)]
        padding[axis] = value
        return padding

    a = np.pad(a, pad_values(-2, (0, 1)), mode='constant')
    b = np.pad(b, pad_values(-2, (1, 0)), mode='constant')
    lapped = a + b
    lapped = lapped.reshape(x.shape[0], -1)
    lapped = lapped[..., :-half_win]
    return lapped

if __name__ == '__main__':
    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(8888)

    sr = zounds.SR11025()
    synth = zounds.SineSynthesizer(sr)

    # make two audio files and group into a batch
    # a1 = synth.synthesize(sr.frequency * 16384, [440])
    # a2 = synth.synthesize(sr.frequency * 16384, [880])
    # batch = np.concatenate([a1[None, ...], a2[None, ...]], axis=0)
    batch, _ = next(batch_stream)
    batch = batch.squeeze()

    # perform a sliding window
    batch = np.pad(batch, ((0, 0), (0, 256)), mode='constant')
    batch = zounds.nputil.sliding_window(batch, (batch.shape[0], 512), (batch.shape[0], 256)).transpose(1, 0, 2)

    # analyze
    coeffs = zounds.spectral.functional.mdct(batch)

    # synthesize
    recon = zounds.spectral.functional.imdct(coeffs)
    recon = overlap_add(recon)

    r1 = zounds.AudioSamples(recon[0], sr).pad_with_silence()
    r2 = zounds.AudioSamples(recon[1], sr).pad_with_silence()
    r3 = zounds.AudioSamples(recon[2], sr).pad_with_silence()
    r4 = zounds.AudioSamples(recon[3], sr).pad_with_silence()

    input('Waiting...')
