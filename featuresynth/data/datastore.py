import lmdb
import numpy as np
import zounds
from .data import preprocess_audio
from ..feature import \
    compute_features_batched, compute_features, sr, total_samples
from ..util.datasource import iter_files
import concurrent.futures
from random import choice
from collections import defaultdict
from uuid import uuid4
import librosa


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

    def _interpret_memview(self, memview, channels):
        raw = np.asarray(memview, dtype=np.uint8)
        # reshape to (time, channels)
        arr = raw.view(dtype=np.float32).reshape((-1, channels))
        return arr

    def iter_feature(self, feature, channels):
        for key in self.iter_keys():
            yield self._get_feature_data(key, feature, channels)

    def _random_slice(self, arr, feature_length):

        # choose random start and end indices
        try:
            start = np.random.randint(0, len(arr) - feature_length)
        except ValueError:
            start = 0

        end = start + feature_length

        # Ensure the segment follows (batch, channels, time) convention
        # segment = arr[start: end].copy().T[None, ...]
        segment = self._slice(arr, start, end)

        # TODO: Padding should no longer be necessary, as it's taken care of
        # in the data population step
        if segment.shape[-1] < feature_length:
            diff = feature_length - segment.shape[-1]
            segment = np.pad(
                segment, ((0, 0), (0, 0), (0, diff)), 'constant')
            end += diff

        return segment.astype(np.float32), start, end

    def _slice(self, arr, start, end):
        # Ensure the segment follows (batch, channels, time) convention
        return arr[start:end].copy().T[None, ...]

    def _full_key(self, key, feature):
        return f'{key.decode()}:{feature}'

    def _get(self, key, feature, txn, channels):
        full_key = self._full_key(key, feature)
        memview = txn.get(full_key.encode())
        if memview is None:
            raise KeyError(full_key)
        arr = self._interpret_memview(memview, channels)
        return arr

    def _get_random_aligned_features(
            self,
            key,
            anchor_feature,
            feature_length,
            feature_ratio_spec):
        """
        key - which sound will samples be drawn from?
        anchor_feature - the feature that will be randomly selected, and to
            which other features will be aligned
        feature_length - the number of samples of anchor_feature to draw
        feature_ratio_spec - sample ratios for all other features to be selected
            in the form {
                feature: (feature_size, feature_channels, ratio_to_anchor)
            }
        """

        feature_slices = {}
        anchor_channels = feature_ratio_spec[anchor_feature][1]

        with self.env.begin(write=False, buffers=True) as txn:

            full_anchor_feat = self._get(
                key, anchor_feature, txn, anchor_channels)
            anchor_slice, start, end = self._random_slice(
                full_anchor_feat, feature_length)
            feature_slices[anchor_feature] = anchor_slice

            for feat_name, spec in feature_ratio_spec.items():
                if feat_name == anchor_feature:
                    continue

                size, channels, ratio = spec
                s = start * ratio
                e = end * ratio
                full_feat = self._get(key, feat_name, txn, channels)
                feat_slice = self._slice(full_feat, s, e)
                feature_slices[feat_name] = feat_slice

        return feature_slices

    def batch_stream(self, batch_size, feature_spec, anchor_feature):

        # build ratio spec
        anchor_size, anchor_channels = feature_spec[anchor_feature]
        ratio_spec = {}
        for feat, spec in feature_spec.items():
            size, channels = spec
            ratio_spec[feat] = spec + (size // anchor_size,)

        all_keys = list(self.iter_keys())
        while True:
            batch = defaultdict(list)
            for _ in range(batch_size):
                key = choice(all_keys)
                aligned_features = self._get_random_aligned_features(
                    key, anchor_feature, anchor_size, ratio_spec)
                for feat, slce in aligned_features.items():
                    batch[feat].append(slce)
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
                key = uuid4().hex.encode()
                feature_dict = future.result()
                for feature_name, value in feature_dict.items():
                    full_key = self._full_key(key, feature_name)
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
        samples = preprocess_audio(samples, sr, min_samples=total_samples)

        # features = compute_features_batched(
        #     samples, total_samples, 8, compute_features)

        features = librosa.feature.melspectrogram(
            samples,
            int(sr),
            n_fft=1024,
            hop_length=256,
            n_mels=256)
        features = np.log(features + 1e-12).T.astype(np.float32)

        return {'audio': samples, 'spectrogram': features}
