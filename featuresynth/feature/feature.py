from ..data import cache, LmdbCollection
import librosa
import zounds
import numpy as np
from torch import nn
import torch
from librosa.filters import mel as librosa_mel_fn
from torch.nn import functional as F


class Audio2Mel(nn.Module):
    def __init__(
            self,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            sampling_rate=22050,
            n_mel_channels=80,
            mel_fmin=0.0,
            mel_fmax=None,
    ):
        super().__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################
        window = torch.hann_window(win_length).float()
        mel_basis = librosa_mel_fn(
            sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio):

        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).view(1, 1, -1)

        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (0, p)).squeeze(1)

        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec


data_cache = LmdbCollection('datacache')

# audio_to_mel = Audio2Mel().to(device)

@cache(data_cache)
def audio(file_path, samplerate):
    samples = zounds.AudioSamples.from_file(file_path).mono
    samples = librosa.resample(
        samples, int(samples.samplerate), int(samplerate))
    samples = librosa.util.normalize(samples, axis=-1) * 0.95
    return samples.astype(np.float32)


def normalized_and_augmented_audio(file_path, samplerate):
    samples = audio(file_path, samplerate)[:]
    samples = librosa.util.normalize(samples, axis=-1) * 0.95
    amp = np.random.uniform(low=0.3, high=1.0)
    samples = samples * amp
    return samples

audio_to_mel_22050 = Audio2Mel(
    1024, 256, 1024, int(zounds.SR22050()), 128)


@cache(data_cache)
def spectrogram(file_path, samplerate):
    samples = audio(file_path, samplerate)[:]
    spec = audio_to_mel_22050(samples)
    spec = spec.data.cpu().numpy().T.astype(np.float32)
    return spec

def make_spectrogram_func(audio_func, samplerate, n_fft, hop, n_mels):
    atm = Audio2Mel(
        win_length=n_fft,
        hop_length=hop,
        n_fft=n_fft,
        sampling_rate=int(samplerate),
        n_mel_channels=n_mels)

    def func(file_path, samplerate):
        samples = audio_func(file_path, samplerate)
        spec = atm(samples)
        spec = spec.data.cpu().numpy().T.astype(np.float32)
        return spec

    return func