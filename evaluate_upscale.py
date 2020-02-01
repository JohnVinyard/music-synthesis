from collections import defaultdict
import torch
from itertools import cycle
import zounds
from torch import nn
from featuresynth.data import DataStore
from torch.nn import functional as F
from featuresynth.featuregenerator.upscale import Generator, LowResGenerator
from featuresynth.featurediscriminator.upscale import Discriminator, LowResDiscriminator
from featuresynth.util import device
from featuresynth.util.modules import normalize
from torch.optim import Adam
from featuresynth.feature import total_samples, frequency_recomposition, \
    band_sizes, filter_banks, slices, bandpass_filters, sr, feature_channels
from featuresynth.generator import Generator as AudioGenerator
import numpy as np

n_frames = 256
total_feature_samples = total_samples * (n_frames // 64)
noise_dim = 128
feature_channels = 256
initial_dim = 16
channels = 256
batch_size = 8

audio_generator_input_size = 64
audio_generator = AudioGenerator(
    input_size=audio_generator_input_size,
    in_channels=feature_channels,
    channels=128,
    output_sizes=band_sizes,
    filter_banks=filter_banks,
    slices=slices,
    bandpass_filters=bandpass_filters)
audio_generator.load_state_dict(torch.load('generator_dilated.dat'))
[item.fb[0].to(torch.device('cpu')) for item in audio_generator.generators]


def preview(fake_batch):
    # fake_batch = fake_batch * spec_std
    # fake_batch = fake_batch + spec_mean

    fake_batch = torch.from_numpy(fake_batch)
    inp = fake_batch[0]  # (256, 512)

    window = audio_generator_input_size
    inp = inp.unfold(1, window, window)  # (256, 8, 64)

    inp = inp.permute(1, 0, 2)  # (8, 256, 64)
    bands = audio_generator(inp)
    samples = frequency_recomposition(
        [np.concatenate(b.data.cpu().numpy().reshape(1, 1, -1), axis=-1)
         for b in bands.values()],
        total_feature_samples)

    # synth = zounds.MDCTSynthesizer()
    # coeffs = zounds.ArrayWithUnits(fake_batch[0].T, [
    #     zounds.TimeDimension(frequency=sr.frequency * 256, duration=sr.frequency * 512),
    #     zounds.IdentityDimension()
    # ])
    # samples = synth.synthesize(coeffs)

    return zounds.AudioSamples(samples.squeeze(), sr).pad_with_silence()



g = LowResGenerator(
    n_frames,
    256,
    noise_dim,
    initial_dim,
    channels,
    None).initialize_weights().to(device)
g_optim = Adam(g.parameters(), lr=0.0001, betas=(0, 0.9))
g.load_state_dict(torch.load('sequence_gen.dat'))


d = LowResDiscriminator(
    n_frames,
    256,
    channels,
    None,
    None).initialize_weights().to(device)
d_optim = Adam(d.parameters(), lr=0.0001, betas=(0, 0.9))
d.load_state_dict(torch.load('sequence_disc.dat'))

# sizes = cycle(list(g.layers.keys()))
# conditioning = cycle([True, False])


def generator_loss(j):
    return 0.5 * ((j - 1) ** 2).mean()


def disc_loss(r_j, f_j):
    return 0.5 * (((r_j - 1) ** 2).mean() + (f_j ** 2).mean())


def zero_grad():
    g_optim.zero_grad()
    d_optim.zero_grad()
    # for optim in d_optims.values():
    #     optim.zero_grad()
    # for optim in g_optims.values():
    #     optim.zero_grad()


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


def noise(n_examples):
    return torch.FloatTensor(n_examples, noise_dim).normal_(0, 1).to(device)


def generated():
    with torch.no_grad():
        return preview(g(noise(1)).data.cpu().numpy())


def downsample(x, size):
    x = x.view(x.shape[0], 1, -1, n_frames)
    kernel_size = feature_channels // size
    x = \
        F.avg_pool2d(x, (kernel_size, 1),
                     (kernel_size, 1)) * kernel_size

    x = x.view(x.shape[0], -1, n_frames)
    x = normalize(x)
    return x


def get_conditioning(orig, size):
    o = downsample(orig, size)
    c = downsample(orig, size // 4)
    return c, o


results = {}
real = {}
results_u = {}

if __name__ == '__main__':
    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread()

    ds = DataStore('feature_data', '/hdd/musicnet/train_data')

    def train_generator(batch):
        zero_grad()
        freeze(d)
        unfreeze(g)
        fake = g(noise(batch_size))
        f_j = d(fake)
        loss = generator_loss(f_j)
        loss.backward()
        g_optim.step()
        return {'g_loss': loss.item(), 'fake': fake.data.cpu().numpy()}

    def train_discriminator(batch):
        zero_grad()
        freeze(g)
        unfreeze(d)
        fake = g(noise(batch_size))
        f_j = d(fake)
        r_j = d(batch)
        loss = disc_loss(r_j, f_j)
        loss.backward()
        d_optim.step()
        return {'d_loss': loss.item(), 'real': batch.data.cpu().numpy()}

    funcs = cycle([train_discriminator, train_generator])
    td = training_data = {}
    for count, batch in enumerate(ds.batch_stream(batch_size, n_frames)):
        batch = torch.from_numpy(batch).to(device)
        batch = normalize(batch)
        # batch = downsample(batch, 64)
        func = next(funcs)
        training_data.update(func(batch))
        if count > 0 and count % 2 == 0:
            print(f'batch: {count}, g: {{g_loss}}, d: {{d_loss}}'
                  .format(**training_data))

    # for count, batch in enumerate(ds.batch_stream(
    #         batch_size=batch_size, feature_length=n_frames)):
    #     size = next(sizes)
    #     # conditioned = np.random.random() > 0.5
    #     conditioned = True
    #
    #     batch = torch.from_numpy(batch).to(device)
    #
    #     if size == 1:
    #         # there is never any conditioning for loudness time series
    #         conditioned = False
    #
    #         # train discriminator
    #         zero_grad()
    #         freeze(g)
    #         unfreeze(d)
    #         fake_loudness = g.layers[size](noise(batch_size))
    #         real_loudness = downsample(batch, 1)
    #         real[size] = real_loudness.data.cpu().numpy()
    #         r_j = d.layers[size](real_loudness)
    #         f_j = d.layers[size](fake_loudness)
    #         loss = disc_loss(r_j, f_j)
    #         loss.backward()
    #         d_loss = loss.item()
    #         d_optims[size].step()
    #
    #         # train generator
    #         zero_grad()
    #         freeze(d)
    #         unfreeze(g)
    #         fake_loudness = g.layers[size](noise(batch_size))
    #         results[size] = fake_loudness.data.cpu().numpy()
    #         f_j = d.layers[size](fake_loudness)
    #         loss = generator_loss(f_j)
    #         loss.backward()
    #         g_loss = loss.item()
    #         g_optims[size].step()
    #     elif conditioned:
    #         zero_grad()
    #         # train discriminator
    #         freeze(g)
    #         unfreeze(d)
    #         condition, output = get_conditioning(batch, size)
    #         real[size] = condition.data.cpu().numpy(), output.data.cpu().numpy()
    #         fake = g.layers[size](noise(batch_size), condition)
    #         r_f, r_j = d.layers[size](condition, output)
    #         f_f, f_j = d.layers[size](condition, fake)
    #         loss = disc_loss(r_j, f_j)
    #         loss.backward()
    #         d_loss = loss.item()
    #         d_optims[size].step()
    #
    #         # train generator
    #         zero_grad()
    #         freeze(d)
    #         unfreeze(g)
    #         condition, output = get_conditioning(batch, size)
    #         fake = g.layers[size](noise(batch_size), condition)
    #         results[size] = \
    #             condition.data.cpu().numpy(), fake.data.cpu().numpy()
    #         f_f, f_j = d.layers[size](condition, fake)
    #         r_f, r_j = d.layers[size](condition, output)
    #         dist = sum(
    #             [torch.abs(r - f).sum() / r.shape[1] for r, f in zip(r_f, f_f)])
    #         loss = generator_loss(f_j) + dist
    #         g_loss = loss.item()
    #         loss.backward()
    #         g_optims[size].step()
    #     else:
    #         # train discriminator unconditioned
    #         zero_grad()
    #         freeze(g)
    #         unfreeze(d)
    #         _, output = get_conditioning(batch, size)
    #         fake = g.forward(noise(batch_size), output_size=size)
    #         _, f_j = d.layers[size](None, fake)
    #         _, r_j = d.layers[size](None, output)
    #         loss = disc_loss(r_j, f_j)
    #         d_loss = loss.item()
    #         loss.backward()
    #         d_optims[size].step()
    #
    #         # train generator unconditioned
    #         zero_grad()
    #         freeze(d)
    #         unfreeze(g)
    #         fake = g.forward(noise(batch_size), output_size=size)
    #         results_u[size] = \
    #             None, fake.data.cpu().numpy()
    #         _, f_j = d.layers[size](None, fake)
    #         loss = generator_loss(f_j)
    #         g_loss = loss.item()
    #         loss.backward()
    #         g_optims[size].step()
    #
    #     print(
    #         f'batch: {count}, c: {conditioned}, size: {size}, g: {g_loss}, d: {d_loss}')
