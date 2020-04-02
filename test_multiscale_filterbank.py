import zounds
from featuresynth.data import batch_stream
from featuresynth.feature import audio
from featuresynth.audio import RawAudio
from featuresynth.audio.transform import \
    fft_frequency_decompose, fft_frequency_recompose
import torch


def freq_band(start, stop):
    return zounds.FrequencyBand(start, stop)


def make_filter_banks(taps, bands, sr, size):
    out = {}
    for tap, band in zip(taps, bands):
        # KLUDGE: Get rid of this hard-coded value
        if size == 8192:
            start = 0
        else:
            start = sr.nyquist // 2
        stop = sr.nyquist
        fb = zounds.FrequencyBand(start, stop)
        out[size] = zounds.learn.FilterBank(
            sr,
            tap,
            zounds.LinearScale(fb, band),
            0.05,
            normalize_filters=True,
            a_weighting=False)
        print(size, sr, out[size].scale)
        size = size // 2
        sr = sr * 2

    return out


if __name__ == '__main__':

    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)


    path = '/hdd/musicnet/train_data'
    pattern = '*.wav'

    samplerate = zounds.SR22050()
    total_samples = 2 ** 17
    feature_spec = {
        'audio': (2 ** 17, 1)
    }

    feature_funcs = {
        'audio': (audio, (samplerate,))
    }

    batch_size = 1
    bs = batch_stream(
        path, pattern, batch_size, feature_spec, 'audio', feature_funcs)


    filter_banks = make_filter_banks(
        taps=[128] * 5,
        bands=[128] * 5,
        sr=samplerate,
        size=total_samples)

    batch, = next(bs)
    orig = RawAudio(batch, samplerate)

    batch = torch.from_numpy(batch)
    decomposed = fft_frequency_decompose(batch, 8192)
    specs = {k:filter_banks[k].convolve(v) for k, v in decomposed.items()}
    recon = {k:filter_banks[k].transposed_convolve(v) for k, v in specs.items()}
    recomposed = fft_frequency_recompose(recon, total_samples)

    hs = {k:filter_banks[k].temporal_pooling(torch.abs(v), v.shape[-1] // 256, v.shape[-1] // 256 // 2).data.cpu().numpy().squeeze() for k, v in specs.items()}

    r = RawAudio(recomposed.data.cpu().numpy(), samplerate)

    input('Waiting...')

