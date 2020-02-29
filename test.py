import zounds
import numpy as np
from featuresynth.util import device
import torch

if __name__ == '__main__':
    print(zounds.SR11025())
    arr = np.random.normal(0, 1, 100)
    t = torch.from_numpy(arr).to(device)
    print(t)
