import lmdb
import numpy as np
import zounds
from .data import preprocess_audio
from ..feature import compute_features_batched, compute_features, sr
import concurrent.futures
import os
from random import choice


# def audio_filenames(path):
#     for filename in os.listdir(path):
#         key, _ = os.path.splitext(filename)
#         yield os.path.join(path, filename)




# with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
#     work = {
#         executor.submit(func, filename): filename
#         for filename in audio_filenames('/hdd/musicnet/train_data')}
#     for future in concurrent.futures.as_completed(work):
#         filename = work[future]
#         key, _ = os.path.splitext(filename)
#         features = future.result()
#         print(f'completed {key} with feature shape {features.shape}')


class DataStore(object):
    def __init__(self, path, audio_path):
        self.audio_path = audio_path
        self.path = path
        self.env = lmdb.open(
            self.path,
            max_dbs=10,
            map_size=10e10,
            writemap=True,
            map_async=True,
            metasync=True)

    # def __setitem__(self, key, value):
    #     with self.env.begin(write=True, buffers=True) as txn:
    #         txn.put(key, value)
    #
    # def __getitem__(self, key):
    #     with self.env.begin(buffers=True) as txn:
    #         value = txn.get(key)
    #         if value is None:
    #             raise KeyError(key)
    #         return value

    def iter_keys(self):
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key in cursor.iternext(keys=True, values=False):
                yield key

    def batch_stream(self, batch_size, feature_length):
        all_keys = list(self.iter_keys())
        while True:
            batch = []
            for _ in range(batch_size):
                key = choice(all_keys)
                with self.env.begin(write=False, buffers=True) as txn:
                    memview = txn.get(key)
                    raw = np.asarray(memview, dtype=np.uint8)
                    arr = raw.view(dtype=np.float32).reshape(-1, 256)
                    start = np.random.randint(0, len(arr) - feature_length)
                    segment = \
                        arr[start: start + feature_length].copy().T[None, ...]
                    batch.append(segment)

            yield np.concatenate(batch, axis=0)\
                .reshape(batch_size, 256, feature_length)

    def populate(self):
        filenames = [
            os.path.join(self.audio_path, filename)
            for filename in os.listdir(self.audio_path)]

        def func(filename):
            samples = zounds.AudioSamples.from_file(filename)
            samples = preprocess_audio(samples, sr)
            features = compute_features_batched(
                samples, 16384, 8, compute_features)
            return features

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            work = {
                executor.submit(func, filename): filename
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

