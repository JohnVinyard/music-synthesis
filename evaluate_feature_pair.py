from itertools import cycle

import torch
import zounds
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from featuresynth.data import DataStore
from featuresynth.feature import total_samples, frequency_recomposition, band_sizes, filter_banks, slices, bandpass_filters, sr
from featuresynth.featurediscriminator import Discriminator
from featuresynth.featuregenerator import Generator
from featuresynth.util import device
from featuresynth.generator import Generator as AudioGenerator

import numpy as np

batch_size = 8
n_frames = 512
total_feature_samples = total_samples * 8
noise_dim = 128

if __name__ == '__main__':
    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread()

    ds = DataStore('feature_data', '/hdd/musicnet/train_data')

    audio_generator_input_size = 64
    audio_generator = AudioGenerator(
        input_size=audio_generator_input_size,
        in_channels=256,
        channels=128,
        output_sizes=band_sizes,
        filter_banks=filter_banks,
        slices=slices,
        bandpass_filters=bandpass_filters)
    audio_generator.load_state_dict(torch.load('generator_dilated.dat'))
    [item.fb[0].to(torch.device('cpu')) for item in audio_generator.generators]


    def preview(fake_batch):
        fake_batch = torch.from_numpy(fake_batch)
        inp = fake_batch[0]  # (256, 512)

        window = audio_generator_input_size
        inp = inp.unfold(1, window, window)  # (256, 8, 64)

        inp = inp.permute(1, 0, 2)  # (8, 256, 64)
        bands = audio_generator(inp)
        samples = frequency_recomposition(
            [np.concatenate(b.data.cpu().numpy().squeeze(), axis=-1)[None, ...] for b in bands.values()],
            total_feature_samples)
        return zounds.AudioSamples(samples.squeeze(), sr)


    g = Generator(
        frames=n_frames,
        out_channels=256,
        noise_dim=128,
        initial_dim=4,
        channels=128).initialize_weights().to(device)
    g_optim = Adam(g.parameters(), lr=0.0001, betas=(0, 0.9))
    # g.load_state_dict(torch.load('feature_generator.dat'))

    d = Discriminator(
        frames=n_frames,
        feature_channels=256,
        channels=128,
        n_judgements=4).initialize_weights().to(device)
    d_optim = Adam(d.parameters(), lr=0.0001, betas=(0, 0.9))
    # d.load_state_dict(torch.load('feature_disc.dat'))


    def zero_grad():
        g.zero_grad()
        d.zero_grad()


    def set_requires_grad(x, requires_grad):
        if isinstance(x, nn.Module):
            x = [x]
        for item in x:
            for p in item.parameters():
                p.requires_grad = requires_grad


    def freeze(x):
        set_requires_grad(x, False)


    def unfreeze(x):
        set_requires_grad(x, True)


    def train_generator():
        zero_grad()
        freeze(d)
        unfreeze(g)

        noise = \
            torch.FloatTensor(batch_size, noise_dim).normal_(0, 1).to(device)
        fake = g(noise)
        judgements = d(fake)
        loss = (-judgements).mean()
        loss.backward()
        g_optim.step()
        return fake.data.cpu().numpy(), loss.item()


    def train_discriminator(batch):
        zero_grad()
        freeze(g)
        unfreeze(d)

        noise = \
            torch.FloatTensor(batch_size, noise_dim).normal_(0, 1).to(device)
        fake = g(noise)
        fake_judgements = d(fake)

        real_judgements = d(batch)

        loss = \
            (F.relu(1 - real_judgements) + F.relu(1 + fake_judgements)).mean()
        loss.backward()
        d_optim.step()
        return loss.item()


    funcs = cycle([train_generator, train_discriminator])

    real = None
    features = None

    for count, batch in enumerate(ds.batch_stream(batch_size, n_frames)):
        real = batch
        batch = torch.from_numpy(batch).to(device)
        features, g_loss = train_generator()
        d_loss = train_discriminator(batch)
        print(f'Batch {count} generator: {g_loss} disc: {d_loss}')
