import zounds
import numpy as np


def spectrogram(x):
    return np.log(np.abs(zounds.spectral.stft(x)))
