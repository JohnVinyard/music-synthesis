from itertools import cycle

import torch
import zounds

from featuresynth.data import DataStore
import featuresynth.experiment
from featuresynth.feature import sr
from featuresynth.util import device
from featuresynth.experiment import Report

import argparse

ds = DataStore('timit', '/hdd/TIMIT', pattern='*.WAV', max_workers=2)


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
    parser.add_argument(
        '--batch-size',
        help='minibatch size',
        type=int,
        default=32)
    parser.add_argument(
        '--report',
        action='store_true')
    parser.add_argument(
        '--report-examples',
        type=int,
        default=5)
    parser.add_argument(
        '--report-source-update-only',
        action='store_true')
    parser.add_argument(
        '--populate',
        action='store_true')
    args = parser.parse_args()

    experiment = getattr(featuresynth.experiment, args.experiment)()
    print('Running:', experiment.__class__)
    print('Batch Size:', args.batch_size)

    torch.backends.cudnn.benchmark = True

    experiment = experiment.to(device)

    if args.populate:
        ds.populate()
    elif args.report:
        report = Report(experiment)
        report.generate(
            ds,
            'spectrogram',
            args.report_examples,
            sr,
            regenerate=not args.report_source_update_only)
    else:
        if args.resume:
            experiment.resume()

        app = zounds.ZoundsApp(globals=globals(), locals=locals())
        app.start_in_thread(8888)

        anchor_feature = 'spectrogram'
        if args.overfit:
            batch_stream = cycle([next(ds.batch_stream(
                1, experiment.feature_spec, anchor_feature))])
        else:
            batch_stream = ds.batch_stream(
                args.batch_size, experiment.feature_spec, anchor_feature)

        steps = cycle([
            experiment.discriminator_trainer,
            experiment.generator_trainer
        ])

        batch_count = 0
        for batch in batch_stream:

            samples, features = experiment.preprocess_batch(batch)

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
