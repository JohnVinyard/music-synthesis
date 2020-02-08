import zounds
from featuresynth.data import DataStore
from featuresynth.feature import \
    sr, total_samples, frequency_recomposition, feature_channels, band_sizes, \
    filter_banks, bandpass_filters, slices, frequency_decomposition
import numpy as np
from featuresynth.util import device
import torch
from itertools import cycle
from torch.optim import Adam
from featuresynth.generator import Generator
from featuresynth.discriminator import Discriminator
import argparse
from torch.nn import functional as F
from featuresynth.util.modules import zero_grad, freeze, unfreeze

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path',
        required=True)
    parser.add_argument(
        '--pattern',
        default='*.wav')
    parser.add_argument(
        '--limit',
        type=int,
        required=False,
        default=None)
    parser.add_argument(
        '--batch-size',
        type=int,
        required=False,
        default=2)
    parser.add_argument(
        '--time-generator-batch-size',
        default=8,
        required=False)
    parser.add_argument(
        '--feature-loss',
        action='store_true')
    parser.add_argument(
        '--freeze-discriminator',
        action='store_true',
        default=False)
    parser.add_argument(
        '--freeze-generator',
        action='store_true',
        default=False)
    parser.add_argument(
        '--no-disc-loss',
        action='store_true',
        default=False)
    parser.add_argument(
        '--gen-weights',
        required=False)
    parser.add_argument(
        '--disc-weights',
        required=False)
    parser.add_argument(
        '--populate',
        action='store_true')
    args = parser.parse_args()

    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(8888)

    ds = DataStore('timit', args.path, pattern=args.pattern, max_workers=2)
    if args.populate:
        ds.populate()

    feature = None
    bands = None


    def g_sample():
        recmposed = frequency_recomposition(bands, total_samples)
        index = np.random.randint(0, len(recmposed))
        fake_sample = zounds.AudioSamples(recmposed[index], sr)
        mx = fake_sample.max()
        fake_sample /= fake_sample.max()
        coeffs = np.abs(zounds.spectral.stft(fake_sample))
        return fake_sample, coeffs, mx


    def view_band(index):
        from scipy.signal import resample
        band = bands[index][0].squeeze()
        if len(band) != total_samples:
            band = resample(band, total_samples)
        samples = zounds.AudioSamples(band, sr)
        coeffs = np.abs(zounds.spectral.stft(samples))
        return coeffs

    def hear_band(index):
        from scipy.signal import resample
        band = bands[index][0].squeeze()
        if len(band) != total_samples:
            band = resample(band, total_samples)
        samples = zounds.AudioSamples(band, sr)
        samples /= (samples.max() + 1e-12)
        return samples


    def view_real_band(samples, index):
        from scipy.signal import resample
        band = samples[index][0].data.cpu().numpy().squeeze()
        if len(band) != total_samples:
            band = resample(band, total_samples)
        samples = zounds.AudioSamples(band, sr)
        coeffs = np.abs(zounds.spectral.stft(samples))
        return coeffs

    def hear_real_band(samples, index):
        from scipy.signal import resample
        band = samples[index][0].data.cpu().numpy().squeeze()
        if len(band) != total_samples:
            band = resample(band, total_samples)
        samples = zounds.AudioSamples(band, sr)
        samples /= (samples.max() + 1e-12)
        return samples

    def fake_spec(index):
        band = bands[index].squeeze()
        x = filter_banks[index].convolve(torch.from_numpy(band).to(device))
        return np.abs(x.data.cpu().numpy()[0]).T

    def real_spec(samples, index):
        band = samples[index]
        x = filter_banks[index].convolve(band)
        return np.abs(x.data.cpu().numpy())[0].T


    feature_size = 64
    learning_rate = 0.0001
    generator = Generator(
        input_size=feature_size,
        in_channels=feature_channels,
        channels=128,
        output_sizes=band_sizes,
        filter_banks=filter_banks,
        slices=slices,
        bandpass_filters=bandpass_filters)

    # [item.fb[0].to(torch.device('cpu')) for item in generator.generators]
    #
    # def time_generator():
    #     inp = torch.FloatTensor(
    #         args.time_generator_batch_size, feature_channels, feature_size)\
    #         .normal_(0, 1)
    #
    #     total_audio_time = \
    #         (sr.frequency * total_samples * inp.shape[0]) / zounds.Seconds(1)
    #
    #     start = time.time()
    #     bands = generator(inp)
    #     samples = frequency_recomposition(
    #         [b.data.cpu().numpy().squeeze() for b in bands.values()], total_samples)
    #     stop = time.time()
    #     wall_time = stop - start
    #     print(f'CPU Generated {total_audio_time} seconds of audio in {wall_time} seconds')
    #     return samples
    #
    # time_generator()

    [item.fb[0].to(device) for item in generator.generators]
    generator = generator.to(device)
    generator.initialize_weights()
    if args.gen_weights:
        generator.load_state_dict(torch.load(args.gen_weights))
    g_optim = Adam(generator.parameters(), lr=learning_rate, betas=(0, 0.9))



    disc = Discriminator(
        input_sizes=band_sizes,
        feature_size=feature_size,
        feature_channels=feature_channels,
        channels=128,
        kernel_size=3,
        filter_banks=filter_banks,
        slices=slices).to(device).initialize_weights()
    if args.disc_weights:
        disc.load_state_dict(torch.load(args.disc_weights))
    d_optim = Adam(disc.parameters(), lr=learning_rate, betas=(0, 0.9))


    def train_generator(samples, features):
        zero_grad(d_optim, g_optim)

        freeze(disc)
        unfreeze(generator)
        d = disc

        features = torch.from_numpy(features).to(device)

        # generator returns a dictionary of bands
        fake = generator(features)
        output, fake_features = d(fake)
        bands = {s.shape[-1]: s for s in samples}
        real_output, real_features = d(bands)

        loss = (-output).mean()
        for r_f, f_f in zip(real_features, fake_features):
            r_f = r_f / r_f.shape[1]
            f_f = f_f / f_f.shape[1]
            loss += torch.abs(r_f - f_f).sum()
        loss.backward()
        g_optim.step()


        np_bands = \
            [fake[size].data.cpu().numpy().squeeze() for size in band_sizes]
        return loss.item(), np_bands


    def train_discriminator(samples, features):
        zero_grad(d_optim, g_optim)
        unfreeze(disc)
        freeze(generator)

        features = torch.from_numpy(features).to(device)
        fake = generator(features)
        fake_output, fake_features = disc(fake)

        bands = {s.shape[-1]: s for s in samples}
        real_output, real_features = disc(bands)

        loss = (F.relu(1 - real_output) + F.relu(1 + fake_output)).mean()
        loss.backward()
        d_optim.step()

        return loss.item(), None


    funcs = [train_generator]
    if not args.freeze_discriminator:
        funcs.append(train_discriminator)
    turn = cycle(funcs)



    batch_count = 0
    batch_stream = ds.batch_stream(
        args.batch_size,
        {'audio': total_samples, 'spectrogram': feature_size},
        ['audio', 'spectrogram'],
        {'audio': 1, 'spectrogram': feature_channels}
    )

    def decompose(samples):
        bands = frequency_decomposition(samples, band_sizes)
        return \
            [torch.from_numpy(b.astype(np.float32)).to(device) for b in bands]

    for samples, features in batch_stream:
        samples = decompose(samples)
        feature = features[0]
        f = next(turn)
        loss, b = f(samples, features)
        if b is not None:
            bands = b
        print(batch_count, f.__name__, loss)
        batch_count += 1
