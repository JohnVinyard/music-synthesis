from itertools import cycle

import torch
import zounds

from featuresynth.data import DataStore
import featuresynth.experiment
from featuresynth.util import device
from featuresynth.experiment import Report
from featuresynth.train import training_loop

import numpy as np
import argparse

path = '/hdd/LJSpeech-1.1'
pattern = '*.wav'
ds = DataStore('ljspeech', path, pattern=pattern, max_workers=2)

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
    # parser.add_argument(
    #     '--populate',
    #     action='store_true')
    args = parser.parse_args()

    experiment = getattr(featuresynth.experiment, args.experiment)()
    print('Running:', experiment.__class__)
    print('Batch Size:', args.batch_size)

    torch.backends.cudnn.benchmark = True

    experiment = experiment.to(device)

    # if args.populate:
    #     ds.populate()
    if args.report:
        report = Report(experiment)
        report.generate(
            ds,
            'spectrogram',
            args.report_examples,
            experiment.samplerate,
            regenerate=not args.report_source_update_only)
    else:
        if args.resume:
            experiment.resume()

        app = zounds.ZoundsApp(globals=globals(), locals=locals())
        app.start_in_thread(8888)

        if args.overfit:
            # batch_stream = cycle([next(ds.batch_stream(
            #     1, experiment.feature_spec, experiment.anchor_feature))])
            batch_stream = cycle([next(
                experiment.batch_stream(path, pattern, 1))])
        else:
            # batch_stream = ds.batch_stream(
            #     args.batch_size,
            #     experiment.feature_spec,
            #     experiment.anchor_feature)
            batch_stream = experiment.batch_stream(
                path, pattern, args.batch_size)

        # steps = cycle([
        #     experiment.discriminator_trainer,
        #     experiment.generator_trainer
        # ])

        steps = experiment.training_steps

        # BEGIN NEW TRAINING CODE =============================================
        def log_features(exp, pre, result, iteration, elapsed):
            _, features = pre
            return {'real_spec': features[0].T}


        def log_real_audio(exp, pre, result, iteration, elapsed):
            samples, _ = pre
            return {'real': experiment.from_audio(samples, exp.samplerate)}


        def log_loss(exp, pre, result, iteration, elapsed):
            # Ugh, this is a bad heuristic, but probably OK for now to identify
            # loss values
            return {k: v for k, v in result.items() if isinstance(v, float)}


        def log_fake_audio(exp, pre, result, iteration, elapsed):
            try:
                return {
                    'fake': experiment.audio_representation(result['fake'], exp.samplerate)
                }
            except KeyError:
                return None


        def log_fake_sequence(exp, pre, result, iteration, elapsed):
            """
            Generate longer sequences than we're training on
            """

            # TODO: Use elapsed time instead
            if iteration % 1000 != 0:
                return None

            bs = exp.batch_stream(
                path=path,
                pattern=pattern,
                batch_size=1,
                feature_spec={
                    'audio': (32768, 1),
                    'spectrogram': (128, experiment.feature_channels)
                })
            batch = next(bs)
            batch = exp.preprocess_batch(batch)
            samples, features = batch
            real = exp.from_audio(samples, exp.samplerate)
            tensor = torch.from_numpy(features).to(device)
            fake = exp.generator(tensor).data.cpu().numpy()
            fake = exp.audio_representation(fake, exp.samplerate)
            return {'real_seq': real, 'fake_seq': fake}


        tl = training_loop(batch_stream, experiment, device, [
            log_features,
            log_real_audio,
            log_loss,
            log_fake_audio,
            log_fake_sequence
        ])

        for i, elapsed_time, log_results in tl:
            # TODO: This should be pushed down into training_loop somehow
            scalars = \
                {k: v for k, v in log_results.items() if isinstance(v, float)}
            arrs = {
                k: v for k, v in log_results.items()
                if not isinstance(v, float)
                }

            # TODO: This is gross, but it works
            for k, v in arrs.items():
                locals()[k] = v

            minutes = elapsed_time.total_seconds() / 60
            print(f'Batch: {i}, Time: {minutes}, Loss: {scalars}')
        # END NEW TRAINING CODE ================================================

        # batch_count = 0
        # for batch in batch_stream:
        #
        #     # preprocess batch
        #     samples, features = experiment.preprocess_batch(batch)
        #
        #     # log something
        #     real_spec = features[0].T
        #
        #     # log something
        #     real = experiment.from_audio(samples, sr)
        #
        #     # move batch to GPU
        #     samples = torch.from_numpy(real.data).to(device)
        #     features = torch.from_numpy(features).to(device)
        #
        #     # fetch and execute next training step
        #     step = next(steps)
        #     data = step(samples, features)
        #
        #     # log loss to console
        #     print({k: v for k, v in data.items() if 'loss' in k})
        #
        #     # log something
        #     try:
        #         fake = experiment.audio_representation(data['fake'], sr)
        #     except KeyError:
        #         pass
        #
        #     batch_count += 1
