import torch
from featuresynth.util import device
from featuresynth.feature import \
    filter_banks, frequency_recomposition, total_samples, sr, band_sizes, \
    feature_channels, slices, bandpass_filters
import numpy as np
import zounds
from featuresynth.generator import Generator
from torch.optim import Adam
from featuresynth.data import TrainingData
import argparse
from torch.nn import functional as F
import torch

feature = None
bands = None


def g_sample():
    recmposed = frequency_recomposition(bands, total_samples)
    index = np.random.randint(0, len(recmposed))
    fake_sample = zounds.AudioSamples(recmposed[index], sr)
    fake_sample /= fake_sample.max()
    coeffs = np.abs(zounds.spectral.stft(fake_sample))
    return fake_sample, coeffs


def view_band(index):
    from scipy.signal import resample
    band = bands[index].squeeze()
    band = resample(band, total_samples)
    samples = zounds.AudioSamples(band, sr)
    coeffs = np.abs(zounds.spectral.stft(samples))
    return coeffs


def upsampling_experiment(
        iterations=11,
        activation=lambda x: F.leaky_relu(x, 0.2),
        kernel_size=2,
        stride=2,
        padding=0):

    with torch.no_grad():
        in_channels = 16
        batch_size = 2
        t = torch.FloatTensor(batch_size, in_channels, 8).normal_(0, 1)

        for i in range(iterations):


            kernel = torch.FloatTensor(in_channels, in_channels, kernel_size).normal_(0, 1)
            t = F.conv_transpose1d(t, kernel, stride=stride, padding=padding)

            # kernel = torch.FloatTensor(
            #     in_channels * stride, in_channels, kernel_size).normal_(0, 1)
            # t = F.conv1d(t, kernel, stride=1, padding=1)
            # t = t\
            #     .permute(0, 2, 1)\
            #     .contiguous()\
            #     .view(batch_size, -1, in_channels)\
            #     .permute(0, 2, 1)\
            #     .contiguous()
            t = activation(t)

            print(t.shape)

    return t.data.cpu().numpy()


def analyze_peaks(
        iterations,
        activation=lambda x: F.leaky_relu(x, 0.2),
        kernel_size=2,
        stride=2,
        padding=0):

    a = []
    for _ in range(100):

        result = upsampling_experiment(
            iterations,
            activation=activation,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)

        x = result[0].T
        x = x.sum(axis=1)
        a.append(np.abs(np.fft.rfft(x))[1:])

    x = np.mean(a, axis=0)
    return x, x.std()


def overfit_generator(
        r,
        generator,
        gen_optim,
        do_updates=True,
        batch_size=1,
        noise_feature=False):

    """
    Ensure that the generator can overfit to a single sample

    Note that replacing input features with noise here does not matter, as the
    generator is learning to overfit to a single sample
    """
    bands, features = next(r.batch_stream())

    bands = [b[:batch_size, ...] for b in bands]
    features = features[:batch_size, ...]

    np_features = features
    features = torch.from_numpy(features).float().to(device)

    if noise_feature:
        features.normal_(0, 1)

    spectral = []
    for b, fb in zip(bands, filter_banks):
        x = fb.convolve(b)
        spectral.append(x.view(-1))

    target = torch.cat(spectral)

    while True:
        gen = generator(features)

        # bands = \
        #     [band.data.cpu().numpy().reshape((batch_size, -1)) for band in gen]
        bands = \
            [gen[size].data.cpu().numpy().squeeze() for size in band_sizes]
        yield np_features, bands

        gen = torch.cat(
            [fb.convolve(gen[b]).view(-1) for b, fb in zip(gen, filter_banks)])
        # minimize l1 loss
        loss = torch.sum(torch.abs(target - gen))
        loss.backward()
        if do_updates:
            gen_optim.step()

        generator.zero_grad()
        print(loss.item())


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-feature', action='store_true')
    parser.add_argument('--freeze-generator', action='store_true')
    args = parser.parse_args()

    result = upsampling_experiment()

    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread()
    input('Waiting...')


    # feature_size = 64
    # learning_rate = 0.0001
    # generator = Generator(
    #     input_size=feature_size,
    #     in_channels=feature_channels,
    #     channels=128,
    #     output_sizes=band_sizes,
    #     filter_banks=filter_banks,
    #     slices=slices,
    #     bandpass_filters=bandpass_filters).to(device)
    # generator.initialize_weights()
    # gen_optim = Adam(
    #     generator.parameters(), lr=learning_rate, betas=(0, 0.9))
    #
    #
    # batch_size = 2
    # td = TrainingData(
    #     '/hdd/musicnet/train_data',
    #     batch_size=batch_size,
    #     total_samples=total_samples,
    #     sr=sr)
    #
    # for f, b in overfit_generator(
    #         td,
    #         generator,
    #         gen_optim,
    #         noise_feature=args.noise_feature,
    #         do_updates=not args.freeze_generator):
    #
    #     feature = f
    #     bands = b
    # input('Waiting...')
