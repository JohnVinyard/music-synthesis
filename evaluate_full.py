import zounds
from torch.nn import functional as F
from featuresynth.generator.full import MelGanGenerator, DDSPGenerator, TwoDimDDSPGenerator
from featuresynth.discriminator.full import FullDiscriminator, FilterBankDiscriminator
from featuresynth.data import DataStore
from featuresynth.util.modules import \
    least_squares_disc_loss, least_squares_generator_loss, zero_grad, freeze, \
    unfreeze, normalize
from featuresynth.util.display import spectrogram
from featuresynth.feature import sr, total_samples, feature_channels
from featuresynth.util import device
from itertools import cycle
from torch.optim import Adam
import torch
import numpy as np

ds = DataStore('timit', '/hdd/TIMIT', pattern='*.WAV', max_workers=2)

feature_size = 64
batch_size = 4

learning_rate = 1e-4
# g = MelGanGenerator(feature_size, feature_channels).initialize_weights().to(device)
# g = DDSPGenerator(feature_size, feature_channels).initialize_weights().to(device)
g = TwoDimDDSPGenerator(feature_size, feature_channels).initialize_weights().to(device)
g_optim = Adam(g.parameters(), lr=learning_rate, betas=(0, 0.9))

d = FullDiscriminator().initialize_weights().to(device)
# d = FilterBankDiscriminator().initialize_weights().to(device)
d_optim = Adam(d.parameters(), lr=learning_rate, betas=(0, 0.9))

feature_loss_scale = 1

def train_generator(samples, features):
    zero_grad(g_optim, d_optim)
    freeze(d)
    unfreeze(g)

    fake = g(features)
    f_features, f_score = d(fake)
    r_features, r_score = d(samples)

    loss = least_squares_generator_loss(f_score)
    # loss = (-f_score).mean()
    # loss = 0
    feature_loss = 0
    for f_f, r_f in zip(f_features, r_features):
        feature_loss += torch.abs(f_f - r_f).sum() / f_f.contiguous().view(batch_size, -1).shape[-1]
    loss = loss + (feature_loss_scale * feature_loss)
    loss.backward()
    g_optim.step()
    return {'g_loss': loss.item(), 'fake': fake.data.cpu().numpy()}


def train_discriminator(samples, features):
    zero_grad(g_optim, d_optim)
    freeze(g)
    unfreeze(d)

    fake = g(features)
    f_features, f_score = d(fake)
    r_features, r_score = d(samples)

    loss = least_squares_disc_loss(r_score, f_score)
    # loss = (F.relu(1 - r_score) + F.relu(1 + f_score)).mean()
    loss.backward()
    d_optim.step()
    return {'d_loss': loss.item()}

steps = cycle([
    train_discriminator,
    train_generator
])

if __name__ == '__main__':
    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(8888)

    batch_stream = ds.batch_stream(
        batch_size,
        {'audio': total_samples, 'spectrogram': feature_size + 32},
        ['audio', 'spectrogram'],
        {'audio': 1, 'spectrogram': feature_channels})
    # batch_stream = cycle([next(batch_stream)])
    batch_count = 0

    generated = None
    real_spec = None
    real_audio = None

    def fake_audio():
        samples = zounds.AudioSamples(generated[0].squeeze(), sr)
        return samples.pad_with_silence()

    def fake_spec():
        return spectrogram(fake_audio())

    def r_spec():
        return spectrogram(real_audio)

    for samples, features in batch_stream:

        samples /= np.abs(samples).max(axis=-1, keepdims=True) + 1e-12
        features /= features.max(axis=(1, 2), keepdims=True) + 1e-12

        real_spec = features[0].T
        real_audio = zounds.AudioSamples(
            samples[0].squeeze(), sr).pad_with_silence()


        samples = torch.from_numpy(samples).to(device)
        # samples = normalize(samples)

        features = torch.from_numpy(features).to(device)
        # features = normalize(features)

        step = next(steps)
        data = step(samples, features)
        print({k: v for k, v in data.items() if 'loss' in k})
        try:
            generated = data['fake']
        except KeyError:
            pass
        batch_count += 1
