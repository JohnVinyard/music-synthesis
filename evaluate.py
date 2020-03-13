from itertools import cycle

import torch
import zounds

import featuresynth.experiment
from featuresynth.util import device
from featuresynth.experiment import Report
from featuresynth.train import training_loop

import argparse

path = '/hdd/musicnet/train_data'
pattern = '*.wav'

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
        '--prefix',
        default='')
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
    args = parser.parse_args()

    experiment = getattr(featuresynth.experiment, args.experiment)()
    print('Running:', experiment.__class__)
    print('Batch Size:', args.batch_size)
    print('Feature Spec:', experiment.feature_spec)
    print('Inference Feature Spec:', experiment.inference_spec)


    torch.backends.cudnn.benchmark = True

    experiment = experiment.to(device)

    if args.report:
        report = Report(experiment, args.prefix)
        report.generate(
            path,
            pattern,
            args.report_examples,
            regenerate=not args.report_source_update_only)
    else:
        if args.resume:
            experiment.resume(args.prefix)

        app = zounds.ZoundsApp(globals=globals(), locals=locals())
        app.start_in_thread(8888)

        if args.overfit:
            batch_stream = cycle([next(
                experiment.batch_stream(path, pattern, 1))])
        else:
            batch_stream = experiment.batch_stream(
                path, pattern, args.batch_size)


        # steps = experiment.training_steps

        def log_features(exp, pre, result, iteration, elapsed):
            _, features = pre
            return {'real_spec': features[0].T}


        def log_real_audio(exp, pre, result, iteration, elapsed):
            samples, _ = pre
            return {'real': experiment.audio_representation(samples, exp.samplerate)}


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
            if iteration == 0 or iteration % 100 != 0:
                return None

            bs = exp.batch_stream(
                path=path,
                pattern=pattern,
                batch_size=1,
                feature_spec=exp.inference_spec)

            batch = next(bs)
            batch = exp.preprocess_batch(batch)
            samples, features = batch
            real = exp.audio_representation(samples, exp.samplerate)
            tensor = torch.from_numpy(features).to(device)

            fake = exp.generator(tensor)
            try:
                fake = fake.data.cpu().numpy()
            except AttributeError:
                fake = {k:v.data.cpu().numpy() for k,v in fake.items()}

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