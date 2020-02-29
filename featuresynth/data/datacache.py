# from .conjure import cache, LmdbCollection, NoCache
# import librosa
# import zounds
# import numpy as np
#
# # TODO: Move all of this into the feature module
#
# data_cache = LmdbCollection('datacache')
#
#
# @cache(data_cache)
# def audio(file_path, samplerate):
#     # samplerate = zounds.SR11025()
#     samples = zounds.AudioSamples.from_file(file_path).mono
#     if samples.samplerate == samplerate:
#         raise NoCache(samples)
#     samples = librosa.resample(
#         samples, int(samples.samplerate), int(samplerate))
#     return samples.astype(np.float32)
#
#
# @cache(data_cache)
# def spectrogram(file_path, samplerate, n_fft, hop, n_mels):
#     # samplerate = zounds.SR11025()
#     # n_fft = 1024
#     # hop = 256
#     # n_mels = 256
#
#     samples = audio(file_path, samplerate)
#     spec = librosa.feature.melspectrogram(
#         samples,
#         sr=int(samplerate),
#         n_fft=n_fft,
#         hop_length=hop,
#         n_mels=n_mels)
#     spec = np.log(spec + 1e-12)
#     spec = spec.T.astype(np.float32)
#     return spec
#
