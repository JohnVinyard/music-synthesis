import torch
from torch import nn
from torch.nn import functional as F
import zounds
from ..audio.transform import overlap_add


class LappedUpscale(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.stride = kernel_size // 2

        self.new_size = kernel_size * 2

        self.linear = nn.Linear(
            kernel_size * in_channels,
            self.new_size * out_channels)
        self.linear.weight.data.normal_(0, 1)

    def forward(self, x):
        batch, channels, time = x.shape
        x = F.pad(x, (0, self.stride))
        x = x.unfold(-1, self.kernel_size, self.stride)
        # (batch, channels, frames, samples)
        _, _, frames, _ = x.shape
        x = x.permute((0, 2, 1, 3)).contiguous()
        # (batch, frames, channels, samples)
        x = x.view(batch * frames, -1)
        x = self.linear(x)
        # (batch * frames, new_channels * new_time)
        x = x.view(batch, frames, self.out_channels, self.new_size)
        x = x.permute((0, 2, 1, 3)).contiguous()
        x = overlap_add(x, apply_window=True)[..., self.stride:-self.stride]
        return x


def transpose_convolve_upscale(x, n_layers):
    kernel = torch.ones((16, 16, 4)).normal_(0, 1)
    for i in range(n_layers):
        x = F.conv_transpose1d(x, kernel, stride=2, padding=1)
    return x


def lapped_upscale(x, n_layers):
    l = LappedUpscale(16, 16, 4)
    for i in range(n_layers):
        x = l.forward(x)
    return x


if __name__ == '__main__':
    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(9999)
    signal = torch.ones(1, 16, 16).normal_(0, 1)
    tc = transpose_convolve_upscale(signal, 4).data.cpu().numpy()
    lu = lapped_upscale(signal, 4).data.cpu().numpy()
    input('waiting...')

