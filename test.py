import zounds
import numpy as np
from featuresynth.util import device
import torch
import librosa

def check_dilated_conv(signal_size, max_dilation):
    from torch.nn import functional as F
    signal = torch.ones(1, 1, signal_size)
    kernel = torch.ones(1, 1, 2)
    dilation = 1
    while dilation <= max_dilation:
        signal = F.pad(signal, (dilation, 0))
        signal = F.conv1d(signal, kernel, dilation=dilation)
        dilation *= 2
    return signal.data.cpu().numpy().squeeze()

if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9000)
    x = check_dilated_conv(512 + 128, 256)
    input('waiting...')

    # print(torch.__version__)
    # print(zounds.__version__)
    # print(librosa.__version__)
    # print(torch.cuda.is_available())
    # print(zounds.SR11025())
    # arr = np.random.normal(0, 1, 100)
    # t = torch.from_numpy(arr).to(device)
    # print(t)