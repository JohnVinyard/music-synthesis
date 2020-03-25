import torch
from torch import nn
from torch.nn import functional as F


class ComplextSTFTDiscriminator(nn.Module):
    def __init__(
            self,
            window_size,
            hop,
            conditioning_channels=0,
            do_fft=False):

        super().__init__()
        self.do_fft = do_fft
        self.conditioning_channels = conditioning_channels
        self.hop = hop
        self.window_size = window_size
        window = torch.hann_window(self.window_size).float()
        self.register_buffer('window', window)
        channels = 512
        self.main = nn.Sequential(
            nn.Conv1d(
                window_size + 2 + conditioning_channels, channels, 7, 2, 3),
            nn.Conv1d(channels, channels, 7, 2, 3),
            nn.Conv1d(channels, channels, 7, 2, 3),
            nn.Conv1d(channels, channels, 7, 2, 3),
        )
        self.judge = nn.Conv1d(channels, 1, 7, 1, 3)

    def forward(self, x, features):

        if self.do_fft:
            batch, channels, time = x.shape
            x = x.view(batch, time)
            x = torch.stft(
                x, self.window_size, self.hop, self.window_size, self.window)
            # (batch, bins, time, 2)
            batch, bins, time, _ = x.shape
            x = x.permute((0, 1, 3, 2))
            x = x.view(batch, -1, time)
            x = x[:, :, :-1]


        if self.conditioning_channels > 0:
            x = torch.cat([x, features], dim=1)


        features = []
        for layer in self.main:
            x = F.leaky_relu(layer(x), 0.2)
            features.append(x)
        j = self.judge(x)
        return [features], [j]
