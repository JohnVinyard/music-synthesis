import zounds
import numpy as np
import torch


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


def fft_frequency_decompose(x, min_size):
    coeffs = torch.rfft(input=x, signal_ndim=1, normalized=True)

    def make_mask(size, start, stop):
        mask = torch.zeros(size).to(x.device)
        mask[start:stop] = 1
        return mask[None, None, :, None]

    output = {}

    current_size = min_size

    while current_size <= x.shape[-1]:
        sl = coeffs[:, :, :current_size // 2 + 1, :]
        if current_size > min_size:
            mask = make_mask(
                size=sl.shape[2],
                start=current_size // 4,
                stop=current_size // 2 + 1)
            sl = sl * mask
        recon = torch.irfft(
            input=sl,
            signal_ndim=1,
            normalized=True,
            signal_sizes=(current_size,))
        output[recon.shape[-1]] = recon
        current_size *= 2

    return output


def fft_resample(x, desired_size):
    batch, channels, time = x.shape
    coeffs = torch.rfft(input=x, signal_ndim=1, normalized=True)
    # (batch, channels, coeffs, 2)

    new_coeffs_size = desired_size // 2 + 1
    new_coeffs = torch.zeros(batch, channels, new_coeffs_size, 2).to(x.device)
    new_coeffs[:, :, :coeffs.shape[2], :] = coeffs

    samples = torch.irfft(
        input=new_coeffs,
        signal_ndim=1,
        normalized=True,
        signal_sizes=(desired_size,))
    return samples


def fft_frequency_recompose(d, desired_size):
    bands = []
    for band in d.values():
        bands.append(fft_resample(band, desired_size))
    return sum(bands)

if __name__ == '__main__':
    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(9999)

    sr = zounds.SR11025()
    synth = zounds.NoiseSynthesizer(sr)
    noise = synth.synthesize(sr.frequency * 16385)
    signal = torch.from_numpy(noise).view(1, 1, 16384).float()

    bands = fft_frequency_decompose(signal, 512)
    recon = {}

    for k, v in bands.items():
        print(k, v.shape)
        recon[k] = zounds.AudioSamples(
            fft_resample(v, 16384).data.cpu().numpy().squeeze(), sr)

    r = fft_frequency_recompose(bands, 16384)
    r = zounds.AudioSamples(r.data.cpu().numpy().squeeze(), sr)

    input('waiting...')