import zounds
from featuresynth.data import DataStore
from zounds.learn import SincLayer
import torch
from torch.nn import functional as F
from torch import nn
import math

ds = DataStore('timit', '/hdd/TIMIT', pattern='*.WAV', max_workers=2)
batch_stream = ds.batch_stream(1, {
    'audio': (16384, 1),
    'spectrogram': (64, 256)
})

if __name__ == '__main__':
    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(9999)

    sr = zounds.SR11025()
    samples, features = next(batch_stream)
    orig = zounds.AudioSamples(samples.squeeze(), sr)
    samples = torch.from_numpy(samples).float()

    scale = zounds.MelScale(zounds.FrequencyBand(20, sr.nyquist), 128)
    # sinc = SincLayer(scale, 257, sr)
    sinc = zounds.learn.FilterBank(
        sr, 512, scale, 0.9, normalize_filters=True, a_weighting=False)

    spec = sinc.convolve(samples)
    recon = sinc.transposed_convolve(spec)
    ds = F.avg_pool1d(F.relu(spec), 512, 256, 256)

    spec = spec.data.cpu().numpy().squeeze()
    ds = ds.data.cpu().numpy().squeeze()
    fb = sinc.filter_bank.data.cpu().numpy().squeeze()
    recon = zounds.AudioSamples(recon.data.cpu().numpy().squeeze(), sr)
    recon /= recon.max()
    input('Waiting...')
