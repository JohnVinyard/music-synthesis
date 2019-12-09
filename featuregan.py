# import zounds
# import torch
# from torch import nn
# from torch.nn import functional as F
# from torch.nn.init import calculate_gain, xavier_normal_
# from torch.optim import Adam
# import numpy as np
# import scipy
# from threading import Thread
# import os
# from random import choice
# from itertools import cycle
# from time import sleep
# from scipy.fftpack import dct, idct
# from torch.nn.utils import weight_norm
#
# '''
# TODO:
# ==============
#
# Scaled L1 Feature Loss
# -----------------------------------------
# - *try feature loss*
# - try feature loss with normalization via size of feature map
# - try feature loss with l1 distance in instead of cosine distance
# - do tiny batch experiment with feature loss.  Does it take less time to get quality audio?
# - CONCLUSION: Scaled feature loss with l1 distance seems to get to intelligible
#     samples much more quickly
#
#
# Upsampling
# -----------------------------------------------------
# - do tiny batch experiment with different upsampling
# - CONCLUSION: transposed convolutions seem to help reduce noise and get to
#     intelligible samples more quickly
#
# Dilated Generator
# ----------------------------------------------
# CONCLUSION: the dilated generator seems to capture transients better, maybe?
#
#
# Dilated Discriminator
# ------------------------------------------------
# ???
#
#
#
# - is there an issue with the way I'm normalizing features?  What if I don't normalize them?
# - try dilated generator and discriminator
# - try random window discriminators again
# - try sine wave generator
# - intuition: use pooling for the full-resolution discriminators to capture phase
#     *relationships* rather than literal/exact phase
#
# - do tiny batch experiment with chroma, MFCC and low-res spectrogram features
# - do tiny batch experiment with more precise per-channel spectrograms and lower capacity networks
#
#
#
# Once Model is finalized:
# =====================================
# - create lmdb database with audio and features
# - update zounds
# - rewrite deploy script using boto3
# - run on EC2
#
#
# '''
#
#
# def weight_norm(x):
#     return x
#
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# total_samples = 16384
# band_sizes = [1024, 2048, 4096, 8192, total_samples]
#
# feature_channels = 256
# sr = zounds.SR11025()
# band = zounds.FrequencyBand(20, sr.nyquist)
# scale = zounds.GeometricScale(20, sr.nyquist - 20, 0.05, feature_channels)
# taps = 1024
#
# fb = zounds.learn.FilterBank(
#     sr,
#     taps,
#     scale,
#     np.linspace(0.25, 0.5, len(scale)),
#     normalize_filters=False,
#     a_weighting=False).to(device)
#
# chroma_scale = zounds.ChromaScale(band)
# chroma_basis = chroma_scale._basis(scale, zounds.HanningWindowingFunc()).T
#
#
# def generate_filter_banks(band_sizes):
#     n_bands = 128
#     n_taps = 256
#     current_low_freq = 20
#     for i, size in enumerate(band_sizes):
#         ratio = (total_samples / size)
#         new_sr = zounds.SampleRate(
#             sr.frequency * ratio, sr.duration * ratio)
#         freq_band = zounds.FrequencyBand(current_low_freq, new_sr.nyquist)
#         scale = zounds.GeometricScale(
#             freq_band.start_hz, freq_band.stop_hz, 0.05, n_bands)
#         bank = zounds.learn.FilterBank(
#             new_sr,
#             n_taps,
#             scale,
#             # values close to zero get good frequency resolution.  Values close
#             # to one get good frequency resolution
#             0.25,
#             normalize_filters=False,
#             a_weighting=False).to(device)
#         # KLUDGE: What's a principled way to scale this?
#         bank.filter_bank = bank.filter_bank / 100
#         current_low_freq = freq_band.stop_hz
#         yield bank
#
#
# filter_banks = list(generate_filter_banks(band_sizes))
#
#
# def transform(samples):
#     with torch.no_grad():
#         s = torch.from_numpy(samples.astype(np.float32)).to(device)
#         result = fb.convolve(s)
#         result = F.relu(result)
#         result = result.data.cpu().numpy()[..., :total_samples]
#     # result = zounds.log_modulus(result * 10)
#     return result
#
#
# def pooled(result):
#     padding = np.zeros((result.shape[0], result.shape[1], feature_channels))
#     result = np.concatenate([result, padding], axis=-1)
#     result = zounds.sliding_window(
#         result,
#         (result.shape[0], result.shape[1], 512),
#         (result.shape[0], result.shape[1], 256))
#     result = result.max(axis=-1).transpose((1, 2, 0))
#     return result
#
#
# def mfcc(result):
#     result = scipy.fftpack.dct(result, axis=1, norm='ortho')
#     return result[:, 1:13, :]
#
#
# def chroma(result):
#     # result will be (batch, channels, time)
#     batch, channels, time = result.shape
#     result = result.transpose((0, 2, 1)).reshape(-1, channels)
#     result = np.dot(result, chroma_basis)
#     result = result.reshape((batch, time, -1))
#     result = result.transpose((0, 2, 1))
#     return result
#
#
# def low_dim(result, downsample_factor=8):
#     # result will be (batch, channels, time)
#     arr = np.asarray(result)
#     arr = arr.reshape(
#         (result.shape[0], -1, downsample_factor, result.shape[-1]))
#     s = arr.mean(axis=1)
#     return s
#
#
# def compute_features(samples):
#     spectral = transform(samples)
#     p = pooled(spectral)
#     # m = mfcc(p)
#     # ld = low_dim(p, downsample_factor=32)
#     # c = chroma(p)
#     ld = p
#     feature = np.concatenate([ld], axis=1).astype(np.float32)
#     return feature
#
#
# class GeneratorBlock(nn.Module):
#     """
#     Apply a series of increasingly dilated convolutions with optinal upscaling
#     at the end
#     """
#
#     def __init__(self, dilations, channels, kernel_size, upsample_factor=2):
#         super().__init__()
#         self.upsample_factor = upsample_factor
#         self.kernel_size = kernel_size
#         self.channels = channels
#         self.dilations = dilations
#         layers = []
#         for i, d in enumerate(self.dilations):
#             padding = (kernel_size * d) // 2
#             c = nn.Conv1d(
#                 channels,
#                 channels,
#                 kernel_size,
#                 dilation=d,
#                 padding=padding,
#                 bias=False)
#             layers.append(c)
#
#         self.bns = nn.Sequential(
#             *[nn.BatchNorm1d(channels) for _ in self.dilations])
#         self.main = nn.Sequential(*[weight_norm(layer) for layer in layers])
#         # No overlap helps to avoid buzzing/checkerboard artifacts
#         self.upsampler = nn.ConvTranspose1d(channels, channels, 2, 2, 0, bias=False)
#         self.bn = nn.BatchNorm1d(channels)
#
#     def forward(self, x):
#         dim = x.shape[-1]
#         x = x.view(x.shape[0], self.channels, -1)
#         for i, layer in enumerate(self.main):
#             t = layer(x)
#             # TODO: Is the residual the issue?
#             x = F.leaky_relu(x + t[..., :dim], 0.2)
#             x = self.bns[i](x)
#
#         if self.upsample_factor > 1:
#             # x = F.upsample(x, scale_factor=self.upsample_factor, mode='linear')
#             x = self.upsampler(x)
#             x = self.bn(x)
#             x = F.leaky_relu(x, 0.2)
#
#         return x
#
#
# class DilatedChannelJudge(nn.Module):
#     def __init__(
#             self,
#             input_size,
#             channels,
#             features_size,
#             feature_channels,
#             fb):
#
#         super().__init__()
#         self.feature_channels = feature_channels
#         self.features_size = features_size
#         self.channels = channels
#         self.input_size = input_size
#         self.fb = [fb]
#
#         filter_bank_channels = fb.filter_bank.shape[0]
#
#         self.feature_embedding = \
#             nn.Conv1d(feature_channels, channels, 1, 1, 0, bias=False)
#
#         c = channels
#         self.main = nn.Sequential(
#             nn.Conv1d(c + filter_bank_channels, c, 3, dilation=1, padding=1, bias=False),
#             nn.Conv1d(c, c, 3, dilation=3, padding=(9 // 2), bias=False),
#             nn.Conv1d(c, c, 3, dilation=9, padding=(18 // 2), bias=False),
#             nn.Conv1d(c, c, 3, dilation=27, padding=(81 // 2), bias=False),
#
#             nn.Conv1d(c, c, 3, dilation=1, padding=1, bias=False),
#             nn.Conv1d(c, c, 3, dilation=3, padding=(9 // 2), bias=False),
#             nn.Conv1d(c, c, 3, dilation=9, padding=(18 // 2), bias=False),
#             nn.Conv1d(c, c, 3, dilation=27, padding=(81 // 2), bias=False),
#         )
#
#         self.judgement = nn.Conv1d(c, 1, 1, 1, 0, bias=False)
#
#     def forward(self, band, features):
#         batch_size = features.shape[0]
#
#         band = band.view(-1, 1, self.input_size)
#         spectral = self.fb[0].convolve(band)[..., :self.input_size]
#         spectral = F.relu(spectral)
#
#         embedded = F.leaky_relu(self.feature_embedding(features), 0.2)
#
#         features = []
#
#         judgements = []
#
#         x = torch.cat(
#             [spectral, F.upsample(embedded, size=spectral.shape[-1])], dim=1)
#
#         for layer in self.main:
#             x = F.leaky_relu(layer(x), 0.2)
#             features.append(x)
#
#         judgements.append(F.tanh(self.judgement(x)))
#
#         return \
#             torch.cat([j.view(batch_size, -1) for j in judgements], dim=1), \
#             torch.cat([f.view(batch_size, -1) for f in features], dim=1)
#
#
#
# class LowResChannelJudge(nn.Module):
#     def __init__(self, input_size, channels, feature_size, feature_channels,
#                  fb):
#         super().__init__()
#         self.feature_size = feature_size
#         self.feature_channels = feature_channels
#         self.channels = channels
#         self.input_size = input_size
#
#         filter_bank_channels = fb.filter_bank.shape[0]
#
#         self.feature_embedding = \
#             nn.Conv1d(feature_channels, channels, 1, 1, 0, bias=False)
#
#         full = []
#         n_layers = int(np.log2(input_size) - np.log2(feature_size))
#         for i in range(n_layers):
#             full.append(nn.Conv1d(
#                 filter_bank_channels + channels if i == 0 else channels,
#                 channels,
#                 kernel_size=3,
#                 stride=2,
#                 padding=1,
#                 bias=False))
#         self.full = nn.Sequential(*full)
#         self.full_judge = nn.Conv1d(channels, 1, 3, 2, 1, bias=False)
#
#         layers = []
#         layers.append(nn.Conv1d(
#             filter_bank_channels + channels, channels, 1, 1, 0, bias=False))
#         for i in range(int(np.log2(feature_size))):
#             layers.append(nn.Conv1d(
#                 channels,
#                 channels,
#                 kernel_size=3,
#                 stride=2,
#                 padding=1,
#                 bias=False))
#
#         self.main = nn.Sequential(*layers)
#         self.fb = [fb]
#         self.judge = nn.Conv1d(channels, 1, 1, 1, bias=False)
#
#     def forward(self, band, features):
#         batch_size = features.shape[0]
#
#         band = band.view(-1, 1, self.input_size)
#         spectral = self.fb[0].convolve(band)[..., :self.input_size]
#         spectral = F.relu(spectral)
#
#         embedded = F.leaky_relu(self.feature_embedding(features), 0.2)
#
#         features = []
#
#         judgements = []
#
#         x = torch.cat(
#             [spectral, F.upsample(embedded, size=spectral.shape[-1])], dim=1)
#         for layer in self.full:
#             x = F.leaky_relu(layer(x), 0.2)
#             features.append(x / x.shape[1])
#
#         x = F.tanh(self.full_judge(x))
#         judgements.append(x)
#
#         kernel = spectral.shape[-1] // 64
#         low_res = F.avg_pool1d(spectral, kernel, kernel)
#
#         x = torch.cat([low_res, embedded], dim=1)
#         for layer in self.main:
#             x = F.leaky_relu(layer(x), 0.2)
#             features.append(x / x.shape[1])
#
#         judgements.append(F.tanh(self.judge(x)))
#
#         return \
#             torch.cat([j.view(batch_size, -1) for j in judgements], dim=1), \
#             torch.cat([f.view(batch_size, -1) for f in features], dim=1)
#
#
# class ChannelDownsampler(nn.Module):
#     def __init__(
#             self,
#             input_size,
#             target_size,
#             channels,
#             kernel_size,
#             fb,
#             downsample_factor=2):
#
#         super().__init__()
#         self.target_size = target_size
#         self.input_size = input_size
#         self.downsample_factor = downsample_factor
#         self.kernel_size = kernel_size
#         self.channels = channels
#         n_layers = int(np.log2(input_size) - np.log2(target_size))
#         layers = []
#         filter_bank_channels = fb.filter_bank.shape[0]
#         for i in range(n_layers):
#             layers.append(nn.Conv1d(
#                 filter_bank_channels if i == 0 else channels,
#                 channels,
#                 kernel_size,
#                 stride=downsample_factor,
#                 padding=kernel_size // 2,
#                 bias=False))
#
#         self.main = nn.Sequential(*layers)
#         self.fb = [fb]
#
#     def test(self):
#         batch_size = 8
#         inp = torch.FloatTensor(*(batch_size, 1, self.input_size))
#         out, _ = self.forward(inp)
#         print(out.shape)
#         assert (batch_size, self.channels, self.target_size) == out.shape
#
#     def forward(self, x):
#         x = x.view(-1, 1, self.input_size)
#         x = self.fb[0].convolve(x)
#         x = F.relu(x)
#         for layer in self.main:
#             x = F.leaky_relu(layer(x), 0.2)
#         return x
#
#
# class DilatedChannelGenerator(nn.Module):
#     def __init__(self, input_size, target_size, in_channels, channels, fb):
#         super().__init__()
#         self.channels = channels
#         self.in_channels = in_channels
#         self.target_size = target_size
#         self.input_size = input_size
#         self.fb = [fb]
#
#         c = channels
#         fbc = fb.filter_bank.shape[0]
#         self.embedding = nn.Conv1d(in_channels, c, 1, 1, 0, bias=False)
#
#         self.main = nn.Sequential(
#             nn.Conv1d(c, c, 3, dilation=1, padding=1, bias=False),
#             nn.Conv1d(c, c, 3, dilation=3, padding=(9 // 2), bias=False),
#             nn.Conv1d(c, c, 3, dilation=9, padding=(18 // 2), bias=False),
#             nn.Conv1d(c, c, 3, dilation=27, padding=(81 // 2), bias=False),
#
#             nn.Conv1d(c, c, 3, dilation=1, padding=1, bias=False),
#             nn.Conv1d(c, c, 3, dilation=3, padding=(9 // 2), bias=False),
#             nn.Conv1d(c, c, 3, dilation=9, padding=(18 // 2), bias=False),
#             nn.Conv1d(c, c, 3, dilation=27, padding=(81 // 2), bias=False),
#         )
#
#         self.to_samples = nn.Conv1d(c, fbc, 1, 1, 1, bias=False)
#
#         self.bns = nn.Sequential(*[nn.BatchNorm1d(channels) for _ in self.main])
#
#     def forward(self, x):
#         x = x.view(-1, self.in_channels, self.input_size)
#
#         # embed
#         x = F.leaky_relu(self.embedding(x), 0.2)
#         # upsample
#         x = F.upsample(x, size=self.target_size)
#
#         for layer, bn in zip(self.main, self.bns):
#             z = layer(x)
#             z = z[..., :x.shape[-1]].contiguous()
#             x = F.leaky_relu(x + z, 0.2)
#             x = bn(x)
#
#         x = self.to_samples(x)
#         x = self.fb[0].transposed_convolve(x)
#         x = x[..., :self.target_size].contiguous()
#         return x
#
#
# class ChannelGenerator(nn.Module):
#     def __init__(self, input_size, target_size, in_channels, channels, fb):
#         super().__init__()
#         self.channels = channels
#         self.in_channels = in_channels
#         self.target_size = target_size
#         self.input_size = input_size
#
#         self.embedding_layer = \
#             weight_norm(nn.Conv1d(in_channels, channels, 1, 1, bias=False))
#         self.embedding_bn = nn.BatchNorm1d(channels)
#         n_layers = int(np.log2(target_size) - np.log2(input_size))
#         layers = []
#         for _ in range(n_layers):
#             block = GeneratorBlock([1, 3, 9], channels, 3, upsample_factor=2)
#             layers.append(block)
#         self.main = nn.Sequential(*layers)
#
#         filter_bank_channels = fb.filter_bank.shape[0]
#
#         self.to_samples = weight_norm(nn.Conv1d(
#             channels, filter_bank_channels, 1, stride=1, padding=1, bias=False))
#         # KLUDGE: There must be a better way to exclude model parameters?
#         self.fb = [fb]
#
#     def test(self):
#         batch_size = 8
#         inp = np.random.normal(
#             0, 1, (batch_size, self.in_channels, self.input_size))
#         t = torch.from_numpy(inp.astype(np.float32))
#         out = self.forward(t).data.cpu().numpy()
#         print(out.shape)
#         assert (batch_size, 1, self.target_size) == out.shape
#
#     def forward(self, x):
#         x = x.view(-1, self.in_channels, self.input_size)
#         embedded = F.leaky_relu(self.embedding_layer(x), 0.2)
#         embedded = self.embedding_bn(embedded)
#         for i, layer in enumerate(self.main):
#             embedded = layer(embedded)
#         samples = self.to_samples(embedded)
#         samples = self.fb[0].transposed_convolve(samples)
#         return samples[..., :self.target_size]
#
#
# class Generator(nn.Module):
#     def __init__(self, input_size, in_channels, channels, output_sizes):
#         super().__init__()
#         self.channels = channels
#         self.output_sizes = output_sizes
#         self.in_channels = in_channels
#         self.input_size = input_size
#
#         generators = []
#
#         for size, fb in zip(output_sizes, filter_banks):
#             generators.append(
#                 ChannelGenerator(input_size, size, in_channels, channels, fb))
#         self.generators = nn.Sequential(*generators)
#
#     def initialize_weights(self):
#         for name, weight in self.named_parameters():
#             if weight.data.dim() > 2:
#                 if 'samples' in name:
#                     xavier_normal_(weight.data, calculate_gain('tanh'))
#                 else:
#                     xavier_normal_(
#                         weight.data, calculate_gain('leaky_relu', 0.2))
#
#     def test(self):
#         batch_size = 8
#         inp = torch.FloatTensor(
#             *(batch_size, self.in_channels, self.input_size))
#         out = self.forward(inp)
#         assert len(self.output_sizes) == len(out)
#         for item, size in zip(out, self.output_sizes):
#             print(item.shape)
#             assert (batch_size, 1, size) == item.shape
#
#     def forward(self, x):
#         x = x.view(-1, self.in_channels, self.input_size)
#         bands = [g(x) for g in self.generators]
#         return bands
#
#
# class BasicDiscriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.dummy = nn.Parameter(torch.FloatTensor(1))
#
#     def initialize_weights(self):
#         return self
#
#     def forward(self, bands, features):
#         return torch.cat(
#             [fb.convolve(b).view(-1) for b, fb in zip(bands, filter_banks)])
#
#
# class Discriminator(nn.Module):
#     def __init__(
#             self,
#             input_sizes,
#             feature_size,
#             feature_channels,
#             channels,
#             kernel_size):
#         super().__init__()
#         self.kernel_size = kernel_size
#         self.channels = channels
#         self.feature_channels = feature_channels
#         self.feature_size = feature_size
#         self.input_sizes = input_sizes
#
#         self.items = nn.Sequential(
#             *[LowResChannelJudge(size, channels, feature_size, feature_channels,
#                                  fb)
#               for size, fb in zip(input_sizes, filter_banks)])
#
#     def initialize_weights(self):
#         for name, weight in self.named_parameters():
#             if weight.data.dim() > 2:
#                 if 'judge' in name:
#                     xavier_normal_(weight.data, calculate_gain('tanh'))
#                 else:
#                     xavier_normal_(
#                         weight.data, calculate_gain('leaky_relu', 0.2))
#         return self
#
#     def test(self):
#         batch_size = 8
#         bands = [
#             torch.FloatTensor(*(batch_size, 1, size))
#             for size in self.input_sizes]
#         features = torch.FloatTensor(
#             *(batch_size, self.feature_channels, self.feature_size))
#         out = self.forward(bands, features)
#         print(out.shape)
#
#     def forward(self, bands, features):
#         batch_size = features.shape[0]
#         features = features.view(-1, self.feature_channels, self.feature_size)
#
#         judgements = []
#         feat = []
#         for layer, band in zip(self.items, bands):
#             j, f = layer(band, features)
#             judgements.append(j)
#             feat.append(f)
#
#         return \
#             torch.cat([j.view(batch_size, -1) for j in judgements], dim=1), \
#             torch.cat([f.view(batch_size, -1) for f in feat], dim=1)
#
#
# class AudioReservoir(Thread):
#     def __init__(self, path, reservoir):
#         super().__init__(daemon=True)
#         self.reservoir = reservoir
#         self.path = path
#         self.files = os.listdir(self.path)
#
#     def _audio_segment(self):
#         filename = choice(self.files)
#         fullpath = os.path.join(self.path, filename)
#         samples = zounds.AudioSamples.from_file(fullpath).mono
#         samples = zounds.soundfile.resample(samples, sr)
#         _, windowed = samples.sliding_window_with_leftovers(
#             total_samples, 1024, dopad=True)
#         windowed /= (windowed.max(axis=-1, keepdims=True) + 1e-8)
#         # return windowed[100:102, ...]
#         return windowed
#
#     def run(self):
#         while True:
#             windowed = self._audio_segment()
#             self.reservoir.add(windowed)
#             print('Audio reservoir', self.reservoir.percent_full())
#             # break
#
#
# class BatchReservoir(Thread):
#     def __init__(self, reservoir, batch_size, batch_queue):
#         super().__init__(daemon=True)
#         self.batch_queue = batch_queue
#         self.batch_size = batch_size
#         self.reservoir = reservoir
#
#     def _get_batch(self):
#         samples = self.reservoir.get_batch(self.batch_size)
#         features = compute_features(samples)
#         features -= features.mean(axis=(1, 2), keepdims=True)
#         features /= (features.std(axis=(1, 2), keepdims=True) + 1e-8)
#         samples = decompose(samples)
#         return samples, features
#
#     def run(self):
#         while True:
#             if len(self.batch_queue) < 25:
#                 try:
#                     batch = self._get_batch()
#                     self.batch_queue.append(batch)
#                 except ValueError:
#                     sleep(0.1)
#             else:
#                 sleep(0.1)
#                 continue
#
#
# def decompose(samples):
#     bands = frequency_decomposition(samples, band_sizes)
#     return \
#         [torch.from_numpy(b.astype(np.float32)).to(device) for b in bands]
#
#
# class TrainingData(object):
#     def __init__(self, path, batch_size, n_audio_workers=4, n_batch_workers=4):
#         super().__init__()
#         self.n_batch_workers = n_batch_workers
#         self.n_audio_workers = n_audio_workers
#         self.path = path
#         self.batch_size = batch_size
#         self.batch_queue = []
#         self.reservoir = zounds.learn.Reservoir(int(1e5), dtype=np.float32)
#
#         self.audio_workers = \
#             [AudioReservoir(path, self.reservoir) for _ in
#              range(n_audio_workers)]
#         for worker in self.audio_workers:
#             worker.start()
#
#         self.batch_workers = [
#             BatchReservoir(self.reservoir, batch_size, self.batch_queue)
#             for _ in range(n_batch_workers)]
#         for worker in self.batch_workers:
#             worker.start()
#
#     def batch_stream(self):
#         while True:
#             try:
#                 print('batch queue', len(self.batch_queue))
#                 yield self.batch_queue.pop()
#             except IndexError:
#                 print('waiting for batch...')
#                 sleep(1)
#
#
# def frequency_decomposition(samples, sizes):
#     sizes = sorted(sizes)
#     batch_size = samples.shape[0]
#     samples = samples.reshape((-1, samples.shape[-1]))
#     coeffs = dct(samples, axis=-1, norm='ortho')
#     positions = [0] + sizes
#     slices = [
#         slice(positions[i], positions[i + 1])
#         for i in range(len(positions) - 1)]
#     bands = []
#     for size, sl in zip(sizes, slices):
#         new_coeffs = np.zeros((batch_size, size), dtype=np.float32)
#         new_coeffs[:, sl] = coeffs[:, sl]
#         resampled = idct(new_coeffs, axis=-1, norm='ortho')
#         bands.append(resampled)
#     # print('REAL', [band.max() for band in bands])
#     return bands
#
#
# def frequency_recomposition(bands, total_size, include_bands=None):
#     if include_bands is None:
#         include_bands = set(range(len(bands)))
#
#     # print('GENERATED', [band.max() for band in bands])
#     batch_size = bands[0].shape[0]
#     bands = sorted(bands, key=lambda band: len(band))
#     final = np.zeros((batch_size, total_size))
#     for i, band in enumerate(bands):
#         coeffs = dct(band, axis=-1, norm='ortho')
#         new_coeffs = np.zeros((batch_size, total_size))
#         new_coeffs[:, :band.shape[-1]] = coeffs
#         ups = idct(new_coeffs, axis=-1, norm='ortho')
#         if i in include_bands:
#             final += ups
#     return final
#
#
# def test_frequency_decomposition(total_samples, band_sizes):
#     synth = zounds.SineSynthesizer(sr)
#     samples = synth.synthesize(
#         sr.frequency * total_samples, [55, 110, 220, 440, 880, 1660, 1660 * 2])
#     batch = np.repeat(samples[None, :], 8, axis=0)
#     bands = frequency_decomposition(batch, band_sizes)
#     recomposed = frequency_recomposition(bands, total_samples)
#     recomposed = zounds.AudioSamples(recomposed[0], sr).pad_with_silence()
#     bands = [band[0] for band in bands]
#     return bands, recomposed
#
#
# def test_filter_bank_recon(r, return_spectral=False):
#     samples, _ = next(r.batch_stream())
#     samples = samples[:1, ...]
#
#     samples /= samples.max()
#
#     bands = frequency_decomposition(samples, band_sizes)
#     new_bands = []
#     spectral = []
#
#     for band, fb in zip(bands, filter_banks):
#         band = torch.from_numpy(band).float().to(device)
#         sp = fb.convolve(band)
#         spectral.append(sp.data.cpu().numpy())
#         band = fb.transposed_convolve(sp)
#         new_bands.append(band.data.cpu().numpy())
#
#     if return_spectral:
#         return spectral
#
#     final = frequency_recomposition(new_bands, total_samples)
#     orig = zounds.AudioSamples(samples.squeeze(), sr)
#     final = zounds.AudioSamples(final.squeeze(), sr)
#     final /= final.max()
#     return orig, final
#
#
# def overfit_generator(r, generator, gen_optim, do_updates=True, batch_size=1):
#     """
#     Ensure that the generator can overfit to a single sample
#     """
#     bands, features = next(r.batch_stream())
#
#     bands = [b[:batch_size, ...] for b in bands]
#     features = features[:batch_size, ...]
#
#     np_features = features
#     features = torch.from_numpy(features).float().to(device)
#
#     # target = torch.cat([b.view(-1) for b in bands])
#     target = torch.cat(
#         [fb.convolve(b).view(-1) for b, fb in zip(bands, filter_banks)])
#
#     while True:
#         gen = generator(features)
#
#         bands = \
#             [band.data.cpu().numpy().reshape((batch_size, -1)) for band in gen]
#         yield np_features, bands
#
#         # gen = torch.cat([b.contiguous().view(-1) for b in gen])
#         gen = torch.cat(
#             [fb.convolve(b).view(-1) for b, fb in zip(gen, filter_banks)])
#         # minimize l1 loss
#         loss = torch.sum(torch.abs(target - gen))
#         loss.backward()
#         if do_updates:
#             gen_optim.step()
#
#         generator.zero_grad()
#         print(loss.item())
#
#
# def view_channel_spectral(r):
#     bands, _ = next(r.batch_stream())
#     batch_size = len(bands[0])
#     spectral = \
#         [fb.convolve(b).data.cpu().numpy() for b, fb in zip(bands, filter_banks)]
#     bands = [band.data.cpu().numpy().reshape(batch_size, -1) for band in bands]
#     return bands, spectral
#
#
# # DONE: ~Try cosine distance loss in discriminator's feature space~
# # DONE: Do a sanity check with frequency decomposition and then filter_bank inverse
# # normalizing filters and adding A-weighting added a lot of distortion to recon
# # DONE: *Try many smaller discriminators that look at a subset of the output*
# # - this doesn't seem to make a substantial difference
#
#
# # DONE: check gradients in discriminator and try batch norm
#
#
# # DONE : try overfitting on a small *batch* of audio samples
# #   - this still works, giving reasonable reconstructions of four audio samples
#
# # TODO: Look at filter banks again, filter norms and spectrogram of reconstruction
# # TODO: Try l2 distance in discriminator's feature space (include first layer of frozen filter weights)
# # TODO: *try spectrograms as features*
#
#
# # TODO: Get an lmdb database prepped with all audio and features
# # TODO: add weights to the discriminator to look at relative band loudness
# # TODO: Try embedding features for each layer of generator
#
# if __name__ == '__main__':
#     app = zounds.ZoundsApp(globals=globals(), locals=locals())
#     app.start_in_thread(8888)
#
#     batch_size = 2
#     r = TrainingData(
#         '/hdd/musicnet/train_data',
#         batch_size,
#         n_audio_workers=2,
#         n_batch_workers=4)
#
#     feature = None
#     bands = None
#
#
#     def g_sample():
#         recmposed = frequency_recomposition(bands, total_samples)
#         index = np.random.randint(0, len(recmposed))
#         fake_sample = zounds.AudioSamples(recmposed[index], sr)
#         fake_sample /= fake_sample.max()
#         coeffs = np.abs(zounds.spectral.stft(fake_sample))
#         return fake_sample, coeffs
#
#
#     def view_band(index):
#         from scipy.signal import resample
#         band = bands[index][0].squeeze()
#         band = resample(band, total_samples)
#         samples = zounds.AudioSamples(band, sr)
#         coeffs = np.abs(zounds.spectral.stft(samples))
#         return coeffs
#
#
#     # bands, spectral = view_channel_spectral(r)
#     # input('Waiting...')
#     # exit()
#
#
#     # spectral = test_filter_bank_recon(r, return_spectral=True)
#     # input('Waiting...')
#     # exit()
#
#     feature_size = 64
#     learning_rate = 0.0001
#     generator = Generator(
#         input_size=feature_size,
#         in_channels=feature_channels,
#         channels=128,
#         output_sizes=band_sizes).to(device)
#     generator.initialize_weights()
#     gen_optim = Adam(
#         generator.parameters(), lr=learning_rate, betas=(0, 0.9))
#
#     # for f, b in overfit_generator(
#     #         r, generator, gen_optim, do_updates=True, batch_size=4):
#     #     bands = b
#     #     feature = f
#
#     discs = [
#         Discriminator(
#             input_sizes=band_sizes,
#             feature_size=feature_size,
#             feature_channels=feature_channels,
#             channels=128,
#             kernel_size=3).to(device)
#     ]
#     discs = [d.to(device).initialize_weights() for d in discs]
#     d_optims = \
#         [Adam(d.parameters(), lr=learning_rate, betas=(0, 0.9)) for d in discs]
#
#
#     # disc.initialize_weights()
#     #
#     # disc_optim = Adam(
#     #     disc.parameters(), lr=0.0001, betas=(0, 0.9))
#
#
#     def sanity_check():
#         samples, _ = next(r.batch_stream())
#         samples = samples[:1]
#         bands = frequency_decomposition(samples, band_sizes)
#         recon = frequency_recomposition(bands, total_samples)
#         return \
#             zounds.AudioSamples(samples[0], sr), \
#             zounds.AudioSamples(recon[0], sr), \
#             [b[0] for b in bands]
#
#
#     def train_generator(samples, features):
#         index = np.random.randint(0, len(discs))
#         disc = discs[index]
#         print(f'training with disc {index}')
#         generator.zero_grad()
#         disc.zero_grad()
#
#         for p in disc.parameters():
#             p.requires_grad = False
#
#         for p in generator.parameters():
#             p.requires_grad = True
#
#         features = torch.from_numpy(features).to(device)
#         fake = generator(features)
#         output, fake_features = disc(fake, features)
#
#         # bands = decompose(samples)
#         bands = samples
#         real_output, real_features = disc(bands, features)
#
#         loss = \
#             torch.abs(real_output - output).mean() \
#             + torch.sum(torch.abs(fake_features - real_features))
#
#         loss.backward()
#
#         # print('============================================')
#         # for g in zounds.learn.util.gradients(generator):
#         #     print(g)
#
#         gen_optim.step()
#
#         np_bands = [b.data.cpu().numpy().squeeze() for b in fake]
#
#         return loss, np_bands
#
#
#     def train_discriminator(samples, features):
#
#         index = np.random.randint(0, len(discs))
#         disc = discs[index]
#         disc_optim = d_optims[index]
#         print(f'training with disc {index}')
#         generator.zero_grad()
#         disc.zero_grad()
#
#         for p in disc.parameters():
#             p.requires_grad = True
#
#         for p in generator.parameters():
#             p.requires_grad = False
#
#         features = torch.from_numpy(features).to(device)
#         fake = generator(features)
#         fake_output, fake_features = disc(fake, features)
#
#         # bands = decompose(samples)
#         bands = samples
#         real_output, real_features = disc(bands, features)
#
#         loss = -torch.abs(real_output - fake_output).mean()
#         loss.backward()
#
#         disc_optim.step()
#
#         return loss, None
#
#
#     turn = cycle([
#         train_generator,
#         train_discriminator,
#     ])
#
#     for samples, features in r.batch_stream():
#
#         feature = features[0]
#
#         f = next(turn)
#         loss, b = f(samples, features)
#         if b is not None:
#             bands = b
#         # if generated is not None:
#         #     g_sample = generated / generated.max()
#         #     coeffs = np.abs(zounds.spectral.stft(g_sample))
#         #     bands = b
#
#         print(f.__name__, loss.item())
