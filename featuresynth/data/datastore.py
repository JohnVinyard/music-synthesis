import lmdb
import numpy as np
import zounds
from .data import preprocess_audio
from ..feature import compute_features_batched, compute_features, sr
import concurrent.futures
import os
from random import choice


class BaseDataStore(object):
    def __init__(self, path, audio_path):
        super().__init__()
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
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key in cursor.iternext(keys=True, values=False):
                yield key

    def batch_stream(self, batch_size, feature_length):
        raise NotImplemented()

    def _transform_func(self, filename):
        raise NotImplemented()

    def populate(self):
        filenames = [
            os.path.join(self.audio_path, filename)
            for filename in os.listdir(self.audio_path)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            work = {
                executor.submit(self._transform_func, filename): filename
                for filename in filenames}

            for future in concurrent.futures.as_completed(work):
                filename = work[future]
                segments = os.path.split(filename)
                key, _ = os.path.splitext(segments[-1])
                features = future.result()
                print(f'completed {key} with feature shape {features.shape}')
                with self.env.begin(write=True, buffers=True) as txn:
                    txn.put(key.encode(), features.data)
                print(f'wrote {key} with feature shape {features.shape}')


class DataStore(BaseDataStore):
    def __init__(self, path, audio_path):
        super().__init__(path, audio_path)

    def batch_stream(self, batch_size, feature_length):
        all_keys = list(self.iter_keys())
        all_keys = list(filter(lambda x: x == b'2533', all_keys))
        while True:
            batch = []
            for _ in range(batch_size):
                with self.env.begin(write=False, buffers=True) as txn:
                    key = choice(all_keys)
                    memview = txn.get(key)
                    raw = np.asarray(memview, dtype=np.uint8)
                    arr = raw.view(dtype=np.float32).reshape(-1, 256)
                    start = np.random.randint(0, len(arr) - feature_length)
                    segment = \
                        arr[start: start + feature_length].copy().T[None, ...]
                    batch.append(segment)

            yield np.concatenate(batch, axis=0)

    def _transform_func(self, filename):
        samples = zounds.AudioSamples.from_file(filename)
        samples = preprocess_audio(samples, sr)
        features = compute_features_batched(
            samples, 16384, 8, compute_features)
        return features


class MdctDataStore(BaseDataStore):
    def __init__(self, path, audio_path):
        super().__init__(path, audio_path)

    def batch_stream(self, batch_size, feature_length):
        all_keys = list(self.iter_keys())
        while True:
            batch = []
            for _ in range(batch_size):
                with self.env.begin(write=False, buffers=True) as txn:
                    key = choice(all_keys)
                    memview = txn.get(key)
                    raw = np.asarray(memview, dtype=np.uint8)
                    arr = raw.view(dtype=np.float64).reshape(-1, 256)
                    start = np.random.randint(0, len(arr) - feature_length)
                    segment = \
                        arr[start: start + feature_length].copy().T[None, ...]
                    batch.append(segment.astype(np.float32))
            yield np.concatenate(batch, axis=0)

    def _transform_func(self, filename):
        samples = zounds.AudioSamples.from_file(filename)
        samples = preprocess_audio(samples, sr)
        window_sr = zounds.HalfLapped()
        windowed = samples.sliding_window(window_sr)
        windowed = windowed * zounds.OggVorbisWindowingFunc()
        node = zounds.MDCT()
        features = list(node._process(windowed))
        dims = features[0].dimensions
        features = zounds.ArrayWithUnits(np.concatenate(features, axis=0), dims)
        return features


# class DataStore(object):
#     def __init__(self, path, audio_path):
#         self.audio_path = audio_path
#         self.path = path
#         self.env = lmdb.open(
#             self.path,
#             max_dbs=10,
#             map_size=10e10,
#             writemap=True,
#             map_async=True,
#             metasync=True)
#
#     def iter_keys(self):
#         with self.env.begin() as txn:
#             cursor = txn.cursor()
#             for key in cursor.iternext(keys=True, values=False):
#                 yield key
#
#     def batch_stream(self, batch_size, feature_length):
#         all_keys = list(self.iter_keys())
#         while True:
#             batch = []
#             for _ in range(batch_size):
#                 with self.env.begin(write=False, buffers=True) as txn:
#                     # key = choice(all_keys)
#                     key = next(filter(lambda k: k == b'2443', all_keys))
#                     # key = all_keys[0]
#                     memview = txn.get(key)
#                     raw = np.asarray(memview, dtype=np.uint8)
#                     arr = raw.view(dtype=np.float32).reshape(-1, 256)
#                     # padding frames to avoid silence or applause
#                     padding_frames = 256
#                     start = np.random.randint(
#                         padding_frames,
#                         len(arr) - feature_length - padding_frames)
#                     # start = 1024
#                     segment = \
#                         arr[start: start + feature_length].copy().T[None, ...]
#                     batch.append(segment)
#
#             yield np.concatenate(batch, axis=0)
#
#     def populate(self):
#         filenames = [
#             os.path.join(self.audio_path, filename)
#             for filename in os.listdir(self.audio_path)]
#
#         def func(filename):
#             samples = zounds.AudioSamples.from_file(filename)
#             samples = preprocess_audio(samples, sr)
#             features = compute_features_batched(
#                 samples, 16384, 8, compute_features)
#             return features
#
#         with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
#             work = {
#                 executor.submit(func, filename): filename
#                 for filename in filenames}
#
#             for future in concurrent.futures.as_completed(work):
#                 filename = work[future]
#                 segments = os.path.split(filename)
#                 key, _ = os.path.splitext(segments[-1])
#                 features = future.result()
#                 print(f'completed {key} with feature shape {features.shape}')
#                 with self.env.begin(write=True, buffers=True) as txn:
#                     txn.put(key.encode(), features.data)
#                 print(f'wrote {key} with feature shape {features.shape}')
