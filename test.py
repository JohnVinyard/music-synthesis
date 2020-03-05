import zounds
import numpy as np
from featuresynth.util import device
import torch

if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(port=8888)
    print(torch.__version__)
    print(zounds.__version__)
    print(torch.cuda.is_available())
    print(zounds.SR11025())
    arr = np.random.normal(0, 1, 100)
    t = torch.from_numpy(arr).to(device)
    print(t)
    input('Waiting...')