from itertools import cycle

import numpy as np
import torch
import zounds

from featuresynth.data import DataStore
import featuresynth.experiment
from featuresynth.feature import sr
from featuresynth.util import device

import argparse

ds = DataStore('timit', '/hdd/TIMIT', pattern='*.WAV', max_workers=2)

batch_size = 32


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--overfit',
        help='Should generator and discriminator overfit on a single example?',
        action='store_true')
    parser.add_argument(
        '--resume',
        help='Load weights for the models before training',
        action='store_true')
    parser.add_argument(
        '--experiment',
        help='Class name of the experiment to run',
        required=True)
    args = parser.parse_args()

    experiment = getattr(featuresynth.experiment, args.experiment)()
    print('Running:', experiment.__class__)
    experiment = experiment.to(device)

    steps = cycle([
        experiment.discriminator_trainer,
        experiment.generator_trainer
    ])

    if args.resume:
        experiment.resume()

    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(8888)

    if args.overfit:
        batch_stream = cycle(
            [next(ds.batch_stream(1, experiment.feature_spec))])
    else:
        batch_stream = ds.batch_stream(batch_size, experiment.feature_spec)

    batch_count = 0

    for samples, features in batch_stream:
        # normalize samples and features
        samples /= np.abs(samples).max(axis=-1, keepdims=True) + 1e-12
        features /= features.max(axis=(1, 2), keepdims=True) + 1e-12
        real_spec = features[0].T

        real = experiment.from_audio(samples, sr)

        samples = torch.from_numpy(real.data).to(device)
        features = torch.from_numpy(features).to(device)

        step = next(steps)
        data = step(samples, features)
        print({k: v for k, v in data.items() if 'loss' in k})
        try:
            fake = experiment.audio_representation(data['fake'], sr)
        except KeyError:
            pass
        batch_count += 1
