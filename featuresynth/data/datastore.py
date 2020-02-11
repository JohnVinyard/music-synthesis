import lmdb
import numpy as np
import zounds
from .data import preprocess_audio
from ..feature import compute_features_batched, compute_features, sr
from ..util.datasource import iter_files
import concurrent.futures
from random import choice
from collections import defaultdict
from uuid import uuid4


class BaseDataStore(object):
    def __init__(self, path, audio_path, pattern='*.wav', max_workers=4):
        super().__init__()
        self.max_workers = max_workers
        self.pattern = pattern
        self.audio_path = audio_path
        self.path = path
        self.env = lmdb.open(
            self.path,
            max_dbs=10,
            map_size=10e10,
            writemap=True,
            map_async=True,
            metasync=True)

    def iter_keys(self):
        seen = set()
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key in cursor.iternext(keys=True, values=False):
                k, _ = key.split(b':')
                if k not in seen:
                    yield k
                seen.add(k)

    def _get_data(self, key, feature, feature_length, channels):
        full_key = f'{key.decode()}:{feature}'
        with self.env.begin(write=False, buffers=True) as txn:
            memview = txn.get(full_key.encode())
            if memview is None:
                raise KeyError(full_key)
            raw = np.asarray(memview, dtype=np.uint8)
            # reshape to (time, channels)
            arr = raw.view(dtype=np.float32).reshape((-1, channels))
            # choose a random segment
            try:
                start = np.random.randint(0, len(arr) - feature_length)
            except ValueError:
                start = 0
            # Ensure the segment follows (batch, channels, time) convention
            segment = arr[start: start + feature_length].copy().T[None, ...]
            if segment.shape[-1] < feature_length:
                diff = feature_length - segment.shape[-1]
                segment = np.pad(
                    segment, ((0, 0), (0, 0), (0, diff)), 'constant')
            return segment.astype(np.float32)

    def batch_stream(
            self, batch_size, feature_spec):
        all_keys = list(self.iter_keys())
        while True:
            batch = defaultdict(list)
            for _ in range(batch_size):
                key = choice(all_keys)
                for feature, shape in feature_spec.items():
                    length, channels = shape
                    batch[feature].append(self._get_data(
                        key,
                        feature,
                        length,
                        channels))
            finalized = tuple(
                np.concatenate(batch[feature], axis=0)
                for feature in feature_spec)
            yield finalized

    def _transform_func(self, filename):
        raise NotImplemented()

    def populate(self):
        filenames = list(iter_files(self.audio_path, self.pattern))

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers) as executor:
            work = {
                executor.submit(self._transform_func, filename): filename
                for filename in filenames}

            for future in concurrent.futures.as_completed(work):
                key = uuid4().hex
                feature_dict = future.result()
                for feature_name, value in feature_dict.items():
                    full_key = f'{key}:{feature_name}'
                    with self.env.begin(write=True, buffers=True) as txn:
                        feat = np.ascontiguousarray(value, dtype=np.float32)
                        try:
                            txn.put(
                                full_key.encode(), feat.data)
                        except BufferError as be:
                            print(be)
                    print(f'wrote {full_key} with feature shape {feat.shape}')


class DataStore(BaseDataStore):
    def __init__(self, path, audio_path, pattern='*.wav', max_workers=4):
        super().__init__(path, audio_path, pattern, max_workers)

    def _transform_func(self, filename):
        samples = zounds.AudioSamples.from_file(filename)
        total_samples = 16384
        samples = preprocess_audio(samples, sr, min_samples=total_samples)
        features = compute_features_batched(
            samples, total_samples, 8, compute_features)
        return {'audio': samples, 'spectrogram': features}


class MdctDataStore(BaseDataStore):
    def __init__(self, path, audio_path, pattern='*.wav'):
        super().__init__(path, audio_path, pattern)

    def _transform_func(self, filename):
        samples = zounds.AudioSamples.from_file(filename)
        total_samples = 16384
        samples = preprocess_audio(samples, sr, min_samples=total_samples)
        window_sr = zounds.HalfLapped()
        windowed = samples.sliding_window(window_sr)
        windowed = windowed * zounds.OggVorbisWindowingFunc()
        node = zounds.MDCT()
        features = list(node._process(windowed))
        dims = features[0].dimensions
        features = zounds.ArrayWithUnits(np.concatenate(features, axis=0), dims)
        return {'mdct': features}

