from ..util.display import spectrogram
from ..audio.transform import mdct, imdct
import numpy as np
import zounds
import lws
from scipy.signal import hann


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
        return zounds.AudioSamples(self.to_audio()[0], self.samplerate)\
            .pad_with_silence(zounds.Seconds(1))


class RawAudio(BaseAudioRepresentation):
    def __init__(self, data, samplerate):
        super().__init__(data, samplerate)

    @classmethod
    def from_audio(cls, samples, samplerate):
        return cls(samples, samplerate)

    def _reshape(self):
        batch, _, samples = self.data.shape
        return self.data.reshape((batch, samples))

    def to_audio(self):
        return self._reshape()

    def display_features(self):
        return np.array(self._reshape()[0])


class MDCT(BaseAudioRepresentation):
    window_size = 512
    hop_size = 256

    def __init__(self, data, samplerate):
        super().__init__(data, samplerate)

    @classmethod
    def mdct_bins(cls):
        return cls.window_size // 2

    @classmethod
    def from_audio(cls, samples, samplerate):
        # (batch, time, channels) => (batch, channels, time)
        coeffs = mdct(
            samples, cls.window_size, cls.hop_size).transpose((0, 2, 1))
        return cls(coeffs, samplerate)

    def to_audio(self):
        return imdct(self.data.transpose((0, 2, 1)))

    def display_features(self):
        return zounds.log_modulus(np.abs(self.data[0]) * 10).T


class ComplextSTFT(BaseAudioRepresentation):
    window_size = 1024
    hop_size = 256
    channels = window_size // 2 + 1

    proc = lws.lws(window_size, hop_size, mode='music', perfectrec=True)

    def __init__(self, data, samplerate):
        super().__init__(data, samplerate)

    # @classmethod
    # def _batch_stft(cls, x, window, hop):
    #     diff = window - hop
    #     x = np.pad(x, ((0, 0), (0, 0), (0, diff)), mode='constant')
    #     batch, channels, time = x.shape
    #     x = zounds.sliding_window(x, (1, 1, window), (1, 1, hop), flatten=False) \
    #         .reshape((batch, -1, window))
    #     win = hann(window)[None, None, :]
    #     coeffs = np.fft.rfft(x * win, axis=-1)
    #     return coeffs.transpose((0, 2, 1))
    #
    # def _batch_istft(self, x, window, hop):
    #     diff = window - hop
    #     recon = np.fft.irfft(x, axis=1)
    #     win = hann(window)[None, :, None]
    #     recon *= win
    #     batch, _, time = recon.shape
    #     output = np.zeros((batch, 1, time * hop + diff))
    #     for i in range(time):
    #         start = hop * i
    #         stop = start + window
    #         output[:, :, start:stop] += recon[:, :, i][:, None, :]
    #     return output.astype(np.float32)

    @classmethod
    def _batch_stft(cls, x, window, hop):
        coeffs = []
        for item in x:
            s = item.squeeze()
            expected_frames = s.shape[0] // hop
            c = cls.proc.stft(s).T[None, ...]
            coeffs.append(c[..., :expected_frames])
        return np.concatenate(coeffs, axis=0)


    def _batch_istft(self, x, window, hop):
        samples = []
        for item in x:
            s = self.proc.istft(item.T)[None, None, :]
            samples.append(s)
        return np.concatenate(samples, axis=0)

    @property
    def magnitude(self):
        return self.data[:, :self.channels, :]

    @property
    def phase(self):
        return self.data[:, self.channels:, :]

    def to_audio(self):
        batch, channels, time = self.data.shape

        real = self.data[:, :channels // 2, :]
        imag = self.data[:, channels // 2:, :]

        real = np.exp(real)
        imag = np.cumsum(imag, axis=-1)
        imag = (imag + np.pi) % (2 * np.pi) - np.pi

        coeffs = real * np.exp(1j * imag)

        samples = self._batch_istft(coeffs, self.window_size, self.hop_size)

        return samples.reshape((batch, -1)).astype(np.float32)

    @classmethod
    def from_audio(cls, samples, samplerate):
        coeffs = cls._batch_stft(samples, cls.window_size, cls.hop_size)

        real = np.log(np.abs(coeffs) + 1e-12)
        imag = np.unwrap(np.angle(coeffs))
        imag = np.diff(imag, axis=-1)
        imag = np.pad(
            imag, [[0, 0], [0, 0], [1, 0]], mode='constant')

        # two arrays of dimension (batch, channels, time)
        data = np.concatenate([real, imag], axis=1)
        return cls(data, samplerate)


class STFT(BaseAudioRepresentation):
    def __init__(self, data, samplerate):
        super().__init__(data, samplerate)

    def display_features(self):
        return self.display()

    def _inverse_stft(self, coeffs):
        synth = zounds.FFTSynthesizer()
        return synth.synthesize(coeffs)[None, :]

    def to_audio(self):
        return self._inverse_stft(self.data)

    @classmethod
    def _complex_stft(cls, samples, window_sr=None):
        window_sr = window_sr or zounds.HalfLapped()
        return zounds.spectral.stft(samples, window_sr)

    @classmethod
    def from_audio(cls, samples, samplerate):
        return cls(cls._complex_stft(samples), samplerate)


class UnwrappedPhaseSTFT(STFT):
    """
    Adapted from Chris Donahue's IPython notebook here:
    https://colab.research.google.com/drive/10R44MqmTot_bKSdBbyP_KYQ-Elu475eB#scrollTo=Rk6KHaiaz5c9
    """
    proc = lws.lws(1024, 256, perfectrec=True)

    def __init__(self, data, samplerate):
        super().__init__(data, samplerate)

    def to_audio(self):
        log_mag, phase = self.data[..., 0], self.data[..., 1]
        mag = np.exp(log_mag)
        phase = np.cumsum(phase, axis=0)
        phase = (phase + np.pi) % (2 * np.pi) - np.pi
        coeffs = mag * np.exp(1j * phase)
        samples = self.proc.istft(coeffs)
        return zounds.AudioSamples(samples, self.samplerate)[None, :]

    @classmethod
    def from_audio(cls, samples, samplerate):
        coeffs = cls.proc.stft(samples)
        mag = np.abs(coeffs)
        phase = np.angle(coeffs)
        log_mag = np.log(mag)
        u_phase = np.unwrap(phase, axis=0)
        delta_phase = np.diff(u_phase, axis=0)
        delta_phase_padded = np.pad(
            delta_phase, [[1, 0], [0, 0]], mode='constant')
        data = np.stack([log_mag, delta_phase_padded], axis=2)
        return cls(data, samplerate)


class BasePhaseRecovery(BaseAudioRepresentation):
    """
    Adapted from Chris Donahue's IPython notebook here:
    https://colab.research.google.com/drive/10R44MqmTot_bKSdBbyP_KYQ-Elu475eB#scrollTo=Rk6KHaiaz5c9
    """
    proc = lws.lws(1024, 256, perfectrec=True)
    basis = None

    def __init__(self, data, samplerate):
        super().__init__(data, samplerate)

    def to_audio(self):
        coeffs = self.data
        coeffs = np.exp(coeffs)
        pinv = np.linalg.pinv(self.basis)
        mag = np.matmul(coeffs, pinv.T)
        coeffs = self.proc.run_lws(mag)
        samples = self.proc.istft(coeffs)
        return zounds.AudioSamples(samples, self.samplerate)[None, :]

    @classmethod
    def from_audio(cls, samples, samplerate):
        coeffs = cls.proc.stft(samples)
        mag = np.abs(coeffs)
        coeffs = np.matmul(mag, cls.basis.T)
        coeffs = np.log(coeffs)
        return cls(coeffs, samplerate)


sr = zounds.SR11025()
n_bands = 256
mel_scale = zounds.MelScale(zounds.FrequencyBand(20, sr.nyquist - 20), n_bands)
geom_scale = zounds.GeometricScale(20, sr.nyquist - 20, 0.05, n_bands)
linear_scale = zounds.LinearScale(zounds.FrequencyBand(0, sr.nyquist), 513)
mel_scale_basis = mel_scale._basis(linear_scale, zounds.HanningWindowingFunc())
geom_scale_basis = geom_scale._basis(
    linear_scale, zounds.HanningWindowingFunc())


class MelScalePhaseRecover(BasePhaseRecovery):
    basis = mel_scale_basis

    def __init__(self, data, samplerate):
        super().__init__(data, samplerate)


class GeometricScalePhaseRecover(BasePhaseRecovery):
    basis = geom_scale_basis

    def __init__(self, data, samplerate):
        super().__init__(data, samplerate)
