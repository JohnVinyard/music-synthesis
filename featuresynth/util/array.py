import numpy as np


def pad(x, length):
    if len(x) > length:
        raise ValueError(
            f'array already has len {len(x)} but requested length was {length}')

    if len(x) == length:
        return x

    padding = [(0, 0) for _ in range(x.ndim)]
    diff = length - len(x)
    padding[0] = (0, diff)
    x = np.pad(x, padding, mode='constant')
    return x
