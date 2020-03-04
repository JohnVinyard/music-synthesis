import zounds
import numpy as np
from featuresynth.util import device
import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[zounds.ui.AppSettings()])
    args = parser.parse_args()
    app = zounds.ZoundsApp(
        locals=locals(), globals=globals(), secret=args.secret)
    app.start_in_thread(args.port)

    print(torch.__version__)
    print(torch.cuda.is_available())
    print(zounds.SR11025())
    arr = np.random.normal(0, 1, 100)
    t = torch.from_numpy(arr).to(device)
    print(t)

    input('Waiting for user input')
