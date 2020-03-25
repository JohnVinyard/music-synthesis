import numpy as np
from .filesystem import iter_audio_chunks
from ..util import pad
from random import choice
from collections import defaultdict


def random_slice(arr, feature_length):
    try:
        start = np.random.randint(0, len(arr) - feature_length)
    except ValueError:
        start = 0
    end = start + feature_length
    sliced = arr[start: end]
    padded = pad(sliced, feature_length)
    return padded, start, start + feature_length


def batch_stream(
        path,
        pattern,
        batch_size,
        feature_spec,
        anchor_feature,
        feature_funcs):

    # build ratio spec
    anchor_spec = feature_spec[anchor_feature]
    anchor_size, anchor_channels = anchor_spec
    ratio_spec = {}
    for feat, spec in feature_spec.items():
        size, channels = spec
        ratio_spec[feat] = spec + (size // anchor_size,)

    def conform(x, spec):
        size, channels = spec
        return x.T.reshape((-1, channels, size))

    all_file_chunks = list(iter_audio_chunks(path, pattern))
    all_file_chunks = list(filter(lambda x: '2157' in x[0], all_file_chunks))

    while True:
        batch = defaultdict(list)
        batch_count = 0

        while batch_count < batch_size:
            try:
                file_chunk = choice(all_file_chunks)
                func, args = feature_funcs[anchor_feature]

                # this will return a numpy wrapper with an open transaction
                anchor = func(file_chunk, *args)
                anchor, start, end = random_slice(anchor, anchor_size)
                anchor = conform(anchor, anchor_spec)
                batch[anchor_feature].append(anchor)


                # get all other features aligned with this one
                for feat, spec in feature_spec.items():

                    if feat == anchor_feature:
                        continue

                    ratio = ratio_spec[feat][2]
                    ns = start * ratio
                    ne = end * ratio
                    expected_length = ne - ns
                    func, args = feature_funcs[feat]
                    feat_data = func(file_chunk, *args)[ns: ne]
                    padded = pad(feat_data, expected_length)
                    padded = conform(padded, spec)
                    batch[feat].append(padded)
                batch_count += 1
            except Exception as e:
                print(f'WARNING: skipping file chunk {file_chunk} because of {e}')

        finalized = tuple(
            np.concatenate(batch[feature], axis=0)
            for feature in feature_spec)
        yield finalized
