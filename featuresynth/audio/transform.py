import zounds
import numpy as np


def overlap_add(x):
    """
    dimensions of x should be (..., frames, samples)
    """
    win_size = x.shape[-1]
    half_win = win_size // 2
    a = x[..., :half_win]
    b = x[..., half_win:]

    def pad_values(axis, value):
        padding = [(0, 0) for _ in range(x.ndim)]
        padding[axis] = value
        return padding

    a = np.pad(a, pad_values(-2, (0, 1)), mode='constant')
    b = np.pad(b, pad_values(-2, (1, 0)), mode='constant')
    lapped = a + b
    lapped = lapped.reshape(x.shape[0], -1)
    lapped = lapped[..., :-half_win]
    return lapped


def mdct(samples, window, hop):
    """
    samples should be (batch, 1, samples)
    """
    batch_size = samples.shape[0]
    samples = samples.reshape(batch_size, -1)
    batch = np.pad(samples, ((0, 0), (0, hop)), mode='constant')
    windowed = zounds.nputil.sliding_window(
        batch, (batch_size, window), (batch_size, hop))
    # windowed is now (frames, batch, samples)
    windowed = windowed.transpose((1, 0, 2))
    # windowed is now (batch, frames, samples)
    coeffs = zounds.spectral.functional.mdct(windowed)
    return coeffs.astype(np.float32)


def imdct(coeffs):
    recon = zounds.spectral.functional.imdct(coeffs)
    recon = overlap_add(recon)
    return recon