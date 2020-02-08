import zounds
import numpy as np


def spectrogram(x):
    return np.abs(zounds.spectral.stft(x))
