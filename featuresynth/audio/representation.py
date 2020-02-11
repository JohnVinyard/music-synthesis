from ..util.display import spectrogram
from ..audio.transform import mdct, imdct
import numpy as np
import zounds


class BaseAudioRepresentation(object):
    def __init__(self, data, samplerate):
        super().__init__()
        self.samplerate = samplerate
        self.data = data

    @classmethod
    def from_audio(cls, samples, samplerate):
        raise NotImplementedError()

    def to_audio(self):
        raise NotImplementedError()

    def display_features(self):
        raise NotImplementedError()

    def display(self):
        raw = self.to_audio()[0]
        audio = zounds.AudioSamples(raw, self.samplerate)
        return spectrogram(audio)

    def listen(self):
        return zounds.AudioSamples(self.to_audio()[0], self.samplerate)


class RawAudio(BaseAudioRepresentation):
    def __init__(self, data, samplerate):
        super().__init__(data, samplerate)

    @classmethod
    def from_audio(cls, samples, samplerate):
        return cls(samples, samplerate)

    def to_audio(self):
        return self.data

    def display_features(self):
        return np.array(self.data[0])


class MDCT(BaseAudioRepresentation):
    def __init__(self, data, samplerate):
        super().__init__(data, samplerate)

    @classmethod
    def from_audio(cls, samples, samplerate):
        # (batch, time, channels) => (batch, channels, time)
        coeffs = mdct(samples, 512, 256).transpose((0, 2, 1))
        coeffs /= np.abs(coeffs).max(axis=(1, 2), keepdims=True) + 1e-12
        return cls(coeffs, samplerate)

    def to_audio(self):
        return imdct(self.data.transpose((0, 2, 1)))

    def display_features(self):
        return zounds.log_modulus(np.abs(self.data[0]) * 10).T
