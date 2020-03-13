import argparse
import featuresynth.experiment
from featuresynth.util import device
from featuresynth.train import training_loop
import torch
import zounds

path = '/hdd/musicnet/train_data'
pattern = '*.wav'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

    experiment = getattr(featuresynth.experiment, args.experiment)()
    print('Running: ', experiment.__class__)
    print('Batch Size: ', args.batch_size)

    torch.backends.cudnn.benchmark = True
    experiment = experiment.to(device)

    if args.resume:
        experiment.resume()

    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(8888)

    batch_stream = experiment.batch_stream(path, pattern, args.batch_size)

    def log_features(exp, pre, result, iteration, elapsed):
        features, _ = pre
        return {'real_spec': features[0].T}

    def log_fake_features(exp, pre, result, iteration, elapsed):
        try:
            return {'fake': result['fake']}
        except KeyError:
            return None

    def log_loss(exp, pre, result, iteration, elapsed):
        # Ugh, this is a bad heuristic, but probably OK for now to identify
        # loss values
        return {k: v for k, v in result.items() if isinstance(v, float)}

    tl = training_loop(batch_stream, experiment, device, [
        log_features,
        log_fake_features,
        log_loss,
    ])

    def fake_audio():
        return experiment.features_to_audio(fake[:1])

    def real_audio():
        return experiment.features_to_audio(real_spec.T[None, ...])

    # This is copied verbatim from evaluate.py and should be factored into
    # a common location
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