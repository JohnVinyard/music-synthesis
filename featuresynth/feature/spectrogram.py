from torch import nn
from zounds.learn import FilterBank
from torch.nn import functional as F


class BaseSpectrogram(nn.Module):
    def __init__(self, samplerate):
        super().__init__()
        self.samplerate = samplerate

    def forward(self, x):
        raise NotImplementedError()


class FilterBankSpectrogram(BaseSpectrogram):
    def __init__(
            self,
            samplerate,
            taps,
            scale,
            scaling_factors,
            pooling):
        super().__init__(samplerate)
        self.pool_window, self.pool_hop = pooling
        self.fb = FilterBank(
            samplerate=samplerate,
            kernel_size=taps,
            scale=scale,
            scaling_factors=scaling_factors,
            normalize_filters=False,
            a_weighting=False)

    def forward(self, x):
        x = self.fb.convolve(x)
        x = F.relu(x)
        x = F.avg_pool1d(
            x, self.pool_window, self.pool_hop, padding=self.pool_hop)
        return x


class FFTSpectrogram(BaseSpectrogram):
    def __init__(self, samplerate, basis):
        super().__init__(samplerate)
        self.basis = basis

    def forward(self, x):
        raise NotImplementedError()
