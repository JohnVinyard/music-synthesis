from torch import nn
from torch.nn.init import xavier_normal_, calculate_gain
from torch.nn import functional as F
from ..util.modules import UpsamplingStack, LearnedUpSample, UpSample


class FilterBankGenerator(nn.Module):
    def __init__(self, filter_bank):
        super().__init__()
        self._filter_bank = [filter_bank]
        self.main = UpsamplingStack(
            64,
            16384,
            2,
            self._build_layer)
        self.to_frames = nn.Conv1d(256, 128, 7, 1, 3)

    def _build_layer(self, i, curr_size, out_size, first, last):
        return LearnedUpSample(256, 256, 8, 2, lambda x: F.leaky_relu(x, 0.2))

    def to(self, device):
        self.filter_bank.to(device)
        return super().to(device)

    def initialize_weights(self):
        for name, weight in self.named_parameters():
            if weight.data.dim() > 2:
                if 'to_frames' in name:
                    xavier_normal_(weight.data, 1)
                else:
                    xavier_normal_(
                        weight.data, calculate_gain('leaky_relu', 0.2))
        return self

    @property
    def filter_bank(self):
        return self._filter_bank[0]

    def forward(self, x):
        x = self.main(x)
        x = self.to_frames(x)
        x = self.filter_bank.transposed_convolve(x)
        return x
