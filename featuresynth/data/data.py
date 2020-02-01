from threading import Thread
import os
import zounds
from random import choice
from time import sleep
from ..util import device
import numpy as np
import torch
from ..feature import \
    band_sizes, frequency_decomposition, compute_features
from fnmatch import fnmatch


def preprocess_audio(samples, target_samplerate):
    return zounds.soundfile.resample(samples.mono, target_samplerate)


def iter_files(base_path, pattern):
    for dirpath, dirnames, filenames in os.walk(base_path):
        audio_files = filter(
            lambda x: fnmatch(x, pattern),
            (os.path.join(dirpath, fn) for fn in filenames))
        yield from audio_files


class AudioReservoir(Thread):
    def __init__(
            self,
            path,
            reservoir,
            total_samples,
            sr,
            limit_samples=None,
            pattern='*.wav'):

        super().__init__(daemon=True)
        self.limit_samples = limit_samples
        self.sr = sr
        self.total_samples = total_samples
        self.reservoir = reservoir
        self.path = path
        self.files = list(iter_files(path, pattern))
        # self.files = os.listdir(self.path)
        # iter_files(path, pattern)

    def _audio_segment(self):
        filename = choice(self.files)
        fullpath = os.path.join(self.path, filename)

        # samples = zounds.AudioSamples.from_file(fullpath).mono
        # samples = zounds.soundfile.resample(samples, self.sr)

        samples = zounds.AudioSamples.from_file(fullpath)
        samples = preprocess_audio(samples, self.sr)
        _, windowed = samples.sliding_window_with_leftovers(
            self.total_samples, 1024, dopad=True)
        dims = windowed.dimensions

        # scale each segment to have a max of one and exclude silence
        mx = windowed.max(axis=-1)
        indices = np.where(mx > 0)
        orig_len = len(windowed)
        windowed = windowed[indices]
        print(f'Lengths {orig_len} {len(windowed)}')
        windowed /= mx[indices][:, None]
        # windowed /= (windowed.max(axis=-1, keepdims=True) + 1e-8)
        windowed = np.array(windowed)
        indices = np.random.permutation(len(windowed))
        windowed = windowed[indices]
        windowed = zounds.ArrayWithUnits(windowed, dims)
        return windowed

    def run(self):
        while True:
            windowed = self._audio_segment()
            windowed = windowed[:self.limit_samples]
            if self.limit_samples is not None and \
                            len(self.reservoir) >= self.limit_samples:
                break
            self.reservoir.add(windowed)
            print('Audio reservoir', self.reservoir.percent_full())


class BatchReservoir(Thread):
    def __init__(self, reservoir, batch_size, batch_queue):
        super().__init__(daemon=True)
        self.batch_queue = batch_queue
        self.batch_size = batch_size
        self.reservoir = reservoir

    def _get_batch(self):
        samples = self.reservoir.get_batch(self.batch_size)
        features = compute_features(samples)
        samples = decompose(samples)
        return samples, features

    def run(self):
        while True:
            if len(self.batch_queue) < 25:
                try:
                    batch = self._get_batch()
                    self.batch_queue.append(batch)
                except ValueError:
                    sleep(0.1)
            else:
                sleep(0.1)
                continue


def decompose(samples):
    bands = frequency_decomposition(samples, band_sizes)
    return \
        [torch.from_numpy(b.astype(np.float32)).to(device) for b in bands]


class TrainingData(object):
    def __init__(
            self,
            path,
            batch_size,
            total_samples,
            sr,
            n_audio_workers=4,
            n_batch_workers=4,
            limit_samples=None,
            reservoir_size=int(1e5),
            pattern='*.wav'):

        super().__init__()
        self.reservoir_size = reservoir_size
        self.n_batch_workers = n_batch_workers
        self.n_audio_workers = n_audio_workers
        self.path = path
        self.batch_size = batch_size
        self.batch_queue = []
        self.reservoir = zounds.learn.Reservoir(
            self.reservoir_size, dtype=np.float32)

        self.audio_workers = \
            [AudioReservoir(path, self.reservoir, total_samples, sr,
                            limit_samples, pattern=pattern) for _ in
             range(n_audio_workers)]
        for worker in self.audio_workers:
            worker.start()

        self.batch_workers = [
            BatchReservoir(self.reservoir, batch_size, self.batch_queue)
            for _ in range(n_batch_workers)]
        for worker in self.batch_workers:
            worker.start()

    def batch_stream(self):
        while True:
            try:
                print('batch queue', len(self.batch_queue))
                yield self.batch_queue.pop()
            except IndexError:
                print('waiting for batch...')
                sleep(1)
