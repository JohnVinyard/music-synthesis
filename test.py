import zounds
import numpy as np
from featuresynth.util import device
import torch
import librosa

if __name__ == '__main__':
    print(torch.__version__)
    print(zounds.__version__)
    print(librosa.__version__)
    print(torch.cuda.is_available())
    print(zounds.SR11025())
    arr = np.random.normal(0, 1, 100)
    t = torch.from_numpy(arr).to(device)
    print(t)