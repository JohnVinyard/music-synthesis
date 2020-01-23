import torch
from torch import nn
from torch.nn import functional as F


class LappedUpscale(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.stride = kernel_size // 2
        self.new_size = kernel_size * 2
        self.padding_amt = kernel_size // 2
        self.linear = nn.Linear(
            in_channels * kernel_size,
            out_channels * self.new_size,
            bias=False)
        self.linear.weight.data.normal_(0, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        frames = x.shape[-1]
        x = x.view(batch_size, self.in_channels, frames)
        print(x.shape)
        x = F.pad(x, (self.padding_amt, self.padding_amt))
        print(x.shape)
        x = x.unfold(-1, self.kernel_size, self.stride)
        # now we have (batch, channels, n_windows, window_size)
        print(x.shape)

        x = x.permute((0, 2, 1, 3))
        # now we have (batch, n_windows, channels, window_size)
        n_windows = x.shape[1]
        print(x.shape)
        x = x.contiguous().view(batch_size * n_windows, -1)
        # now we have (batch * n_windows, channels * window_size)
        print(x.shape)
        x = self.linear(x)
        print(x.shape)
        x = x.view(batch_size, n_windows, self.out_channels, self.new_size)
        print(x.shape)


        #
        # window = torch.hann_window(self.new_size, periodic=True)
        # x = x * window[None, None, None, :]
        # print(x.shape)
        #
        # # x = x.fold(2, self.new_size, self.new_size // 2)





if __name__ == '__main__':
    t = torch.FloatTensor(8, 16, 128).normal_(0, 1)
    ls = LappedUpscale(16, 16, 4)
    us = ls(t)



