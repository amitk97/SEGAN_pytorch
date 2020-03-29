import numpy as np


def normalize_wave_minmax(x):
    return (2. / 65535.) * (x - 32767.) + 1.


def pre_emphasize(x, coef=0.95):
    if coef <= 0:
        return x
    x0 = np.reshape(x[0], (1,))
    diff = x[1:] - coef * x[:-1]
    concat = np.concatenate((x0, diff), axis=0)
    return concat
