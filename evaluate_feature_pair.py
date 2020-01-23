from itertools import cycle

import torch
import zounds
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from featuresynth.data import DataStore, MdctDataStore
from featuresynth.feature import total_samples, frequency_recomposition, \
    band_sizes, filter_banks, slices, bandpass_filters, sr, feature_channels
from featuresynth.featurediscriminator import FrameDiscriminator as Discriminator
from featuresynth.featuregenerator import FrameGenerator as Generator, AutoEncoder
from featuresynth.util import device
from featuresynth.generator import Generator as AudioGenerator

import numpy as np

batch_size = 4
n_frames = 256
total_feature_samples = total_samples * (n_frames // 64)
noise_dim = 128


if __name__ == '__main__':

    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread()

    ds = DataStore('feature_data', '/hdd/musicnet/train_data')
    # ds = MdctDataStore('mdct_data', '/hdd/musicnet/train_data')


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


    def autoregressive_generation(real=True):
        if real:
            primer = next(ds.batch_stream(1, 256))
            primer = primer / (primer.max() + 1e-8)
            primer = torch.from_numpy(primer).to(device)
        else:
            primer = torch.FloatTensor(1, feature_channels, n_frames).uniform_(0, 1).to(device)
        frames = g.generate(primer, 256)
        frames = frames.data.cpu().numpy()
        return preview(frames)

    ae = AutoEncoder().initialize_weights().to(device)
    # ae.load_state_dict(torch.load('feature_autoencoder.dat'))
    ae_optim = Adam(ae.parameters(), lr=0.0001, betas=(0, 0.9))


    g = Generator(
        frames=n_frames,
        out_channels=feature_channels,
        noise_dim=128,
        initial_dim=16,
        channels=128,
        ae=ae).initialize_weights().to(device)
    g_optim = Adam(g.parameters(), lr=0.0001, betas=(0, 0.9))
    # g.load_state_dict(torch.load('ar_gen.dat'))

    d = Discriminator(
        frames=n_frames,
        feature_channels=feature_channels,
        channels=128,
        n_judgements=1,
        ae=ae).initialize_weights().to(device)
    d_optim = Adam(d.parameters(), lr=0.0001, betas=(0, 0.9))
    # d.load_state_dict(torch.load('ardisc.dat'))



    def zero_grad():
        g_optim.zero_grad()
        d_optim.zero_grad()
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


    def noise_vector(batch_size):
        vec = torch.FloatTensor(batch_size * n_frames, noise_dim).to(device).normal_(0, 1)
        return vec


    def train_generator(batch):
        zero_grad()

        freeze(d)
        unfreeze(g)

        noise = noise_vector(batch_size)
        # noise = F.pad(batch, (1, 0))[:, :, :-1]
        latent, fake = g(noise)
        d_latent, judgements = d(fake)
        # r_latent, r_judgements = d(batch)

        loss = 0.5 * ((judgements - 1)**2).mean()  # + torch.abs(d_latent - r_latent).sum()
        loss.backward()
        g_optim.step()
        return latent.data.cpu().numpy(), fake.data.cpu().numpy(), loss.item()


    def train_discriminator(batch):
        zero_grad()
        freeze(g)
        unfreeze(d)

        noise = noise_vector(batch_size)
        # noise = F.pad(batch, (1, 0))[:, :, :-1]
        latent, fake = g(noise)


        f_latent, fake_judgements = d(fake)
        r_latent, real_judgements = d(batch)


        loss = 0.5 * (((real_judgements - 1) ** 2).mean() + (fake_judgements ** 2).mean())
        loss.backward()
        d_optim.step()

        return \
            loss.item(), \
            real_judgements.data.cpu().numpy(), \
            fake_judgements.data.cpu().numpy(), \
            fake.data.cpu().numpy(), \
            f_latent.data.cpu().numpy(), \
            r_latent.data.cpu().numpy()


    def fractal(x, frame_size=16):
        x = x.squeeze()
        x = x.unfold(1, frame_size, frame_size // 2)
        x = x.contiguous().view(-1, frame_size)
        norm = torch.norm(x, dim=-1, keepdim=True)
        x = x / (norm + 1e-12)
        return x

    def fractal2(x, frame_size):
        x = x.squeeze()
        x = x.unfold(1, frame_size, frame_size // 2)
        while x.shape[-1] > 1:
            yield x.contiguous().view(-1, frame_size)
            x = torch.norm(x, dim=-1)
            frame_size = min(x.shape[-1], frame_size)
            step = max(1, frame_size // 2)
            x = x.unfold(1, frame_size, step)


    def cosine_loss(x, y):
        return (-F.cosine_similarity(x, y, dim=-1)).mean()

    def fractal_loss(x, y):
        losses = [F.mse_loss(x, y)]
        frame_size = 4
        for a, b, in zip(fractal2(x, frame_size), fractal2(y, frame_size)):
            losses.append(cosine_loss(a, b))
        return sum(losses)

    def test_dilations(signal_length, max_dilation):
        x = s = torch.ones(1, 1, signal_length)
        kernel = torch.ones(1, 1, 2)
        dilation = 1
        while dilation <= max_dilation:
            x = F.conv1d(F.pad(x, (dilation, 0)), kernel, dilation=dilation)
            print(x.shape)
            dilation *= 2
        return x.data.cpu().numpy().squeeze()

    funcs = cycle([train_generator, train_discriminator])

    real = None
    features = None
    fake_features = None
    predicted_frame = None
    real_frame = None
    real_j = None
    fake_j = None
    g_latent = None
    d_f_latent = None
    d_r_latent = None

    # stats_batch = next(ds.batch_stream(512, n_frames))
    # stats_batch = (stats_batch / (np.abs(stats_batch).max(axis=(1, 2), keepdims=True) + 1e-12))
    # spec_mean = stats_batch.mean(axis=(0, 2), keepdims=True)
    # spec_std = stats_batch.std(axis=(0, 2), keepdims=True)


    recon_frames = None
    encoding = None
    orig_frames = None

    def listen_to_recon():
        with torch.no_grad():
            orig = next(ds.batch_stream(1, n_frames))
            orig = \
                (orig / (np.abs(orig).max(axis=(1, 2), keepdims=True) + 1e-12))
            # (1, channels, frames)
            orig = torch.from_numpy(orig)\
                .to(device)\
                .permute(0, 2, 1) \
                .contiguous()\
                .view(-1, 1, 256)

            # mx, _ = torch.max(orig, dim=-1, keepdim=True)
            # orig = orig / (mx + 1e-8)

            encoded, recon = ae(orig)
            # recon = recon * mx
            # (n_frames, 1, 256)
            recon = recon.permute(1, 2, 0).data.cpu().numpy()
            return preview(recon)


    # for count, batch in enumerate(ds.batch_stream(batch_size, n_frames)):
    #     batch = \
    #         (batch / (np.abs(batch).max(axis=(1, 2), keepdims=True) + 1e-12))
    #
    #     batch = torch.from_numpy(batch).to(device)
    #     batch = batch.permute(0, 2, 1).contiguous().view(-1, 1, 256)
    #
    #     # mx, _ = torch.max(batch, dim=-1, keepdim=True)
    #     # batch = batch / (mx + 1e-8)
    #
    #     orig_frames = batch.data.cpu().numpy().squeeze()
    #
    #     ae_optim.zero_grad()
    #     encoded, recon = ae(batch)
    #     encoding = encoded.data.cpu().numpy().squeeze()
    #     recon_frames = recon.data.cpu().numpy().squeeze()
    #     loss = fractal_loss(recon, batch)
    #     loss.backward()
    #     ae_optim.step()
    #     print(loss.item())


    def batch_to_latent(batch):
        batch = \
            (batch / (np.abs(batch).max(axis=(1, 2), keepdims=True) + 1e-12))

        batch_size, channels, frames = batch.shape

        # mx, _ = batch.max(dim=1, keepdim=True)
        # batch = batch / (mx + 1e-8)

        batch = batch.permute(0, 2, 1).contiguous().view(-1, 1, 256)


        with torch.no_grad():
            encoded = ae\
                .encode(batch)\
                .view(batch_size, frames, ae.latent_dim)\
                .permute(0, 2, 1)


        return encoded

    def latent_to_frames(encoded):
        encoded = torch.from_numpy(encoded).to(device)

        batch, latent, frames = encoded.shape
        encoded = encoded\
            .permute(0, 2, 1)\
            .contiguous()\
            .view(batch * frames, latent, 1)

        with torch.no_grad():
            decoded = ae.decode(encoded)
            # (batch * frames, 1, 256)
            decoded = decoded.view(batch, frames, feature_channels).permute(0, 2, 1)

        return decoded.data.cpu().numpy()

    def listen(encoded):
        return preview(latent_to_frames(encoded))


    for count, batch in enumerate(ds.batch_stream(batch_size * n_frames, 1)):
        # (batch, feature_channels, 1)
        batch = batch.transpose(0, 2, 1)
        # (batch, 1, feature_channels)

        # max one normalization

        # batch = batch.sum(axis=1, keepdims=True)
        batch = (batch / (np.abs(batch).max(axis=(-1), keepdims=True) + 1e-12))

        # feature-wise normalization
        # batch = batch - spec_mean
        # batch = batch / spec_std


        batch = torch.from_numpy(batch).to(device)

        # batch = batch_to_latent(batch)

        real = batch.data.cpu().numpy()


        d_loss, real_j, fake_j, features, d_f_latent, d_r_latent = train_discriminator(batch)
        g_latent, fake_features, g_loss = train_generator(batch)
        print(f'Batch {count} generator: {g_loss} disc: {d_loss}')

