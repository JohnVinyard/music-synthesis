import zounds
from featuresynth.data import DataStore
from featuresynth.util.modules import \
    least_squares_disc_loss, least_squares_generator_loss
from featuresynth.generator import MDCTGenerator
from featuresynth.discriminator import MDCTDiscriminator
from featuresynth.audio import MDCT
from featuresynth.feature import sr, total_samples, feature_channels
from featuresynth.util import device
from featuresynth.train import GeneratorTrainer, DiscriminatorTrainer
from itertools import cycle
from torch.optim import Adam
import torch
import numpy as np

ds = DataStore('timit', '/hdd/TIMIT', pattern='*.WAV', max_workers=2)

feature_size = 64
batch_size = 4
learning_rate = 1e-4

g = MDCTGenerator(feature_channels).initialize_weights().to(device)
g_optim = Adam(g.parameters(), lr=learning_rate, betas=(0, 0.9))

d = MDCTDiscriminator(feature_channels).initialize_weights().to(device)
d_optim = Adam(d.parameters(), lr=learning_rate, betas=(0, 0.9))

g_trainer = GeneratorTrainer(
    g, g_optim, d, d_optim, least_squares_generator_loss)

d_trainer = DiscriminatorTrainer(
    g, g_optim, d, d_optim, least_squares_disc_loss)


steps = cycle([
    d_trainer.train,
    g_trainer.train
])

if __name__ == '__main__':
    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(8888)

    batch_stream = ds.batch_stream(
        batch_size,
        {'audio': total_samples, 'spectrogram': feature_size},
        ['audio', 'spectrogram'],
        {'audio': 1, 'spectrogram': feature_channels})
    batch_count = 0

    fake = None

    for samples, features in batch_stream:

        # normalize samples and features
        samples /= np.abs(samples).max(axis=-1, keepdims=True) + 1e-12
        features /= features.max(axis=(1, 2), keepdims=True) + 1e-12
        real_spec = features[0].T

        real = MDCT.from_audio(samples, sr)

        samples = torch.from_numpy(real.data).to(device)
        features = torch.from_numpy(features).to(device)

        step = next(steps)
        data = step(samples, features)
        print({k: v for k, v in data.items() if 'loss' in k})
        try:
            fake = MDCT(data['fake'], sr)
        except KeyError:
            pass
        batch_count += 1
