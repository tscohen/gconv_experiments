
import numpy as np
import skimage.transform as tf
from scipy.ndimage import shift

# TODO: multithreading


def rotate_transform_batch(x, rotation=None):

    r = np.random.uniform(-0.5, 0.5, size=x.shape[0]) * rotation

    # hack; skimage.transform wants float images to be in [-1, 1]
    factor = np.maximum(np.max(x), np.abs(np.min(x)))
    x = x / factor

    x_out = np.empty_like(x)
    for i in range(x.shape[0]):
        x_out[i, 0] = tf.rotate(x[i, 0], r[i])

    x_out *= factor

    return x_out


D4 = np.array([
    # No flip (det = 1)
    [[1., 0],
     [0, 1.]],
    [[0, -1],
     [1, 0]],
    [[-1, 0],
     [0, -1]],
    [[0, 1],
     [-1, 0]],

    # With flip (det = -1)
    [[-1, 0],
     [0, 1]],
    [[0, -1],
     [-1, 0]],
    [[1, 0],
     [0, -1]],
    [[0, 1],
     [1, 0]]
], dtype=int)


def dihedral_transform_batch(x):

    g = np.random.randint(low=0, high=8, size=x.shape[0])

    h, w = x.shape[-2:]
    hh = (h - 1) / 2.
    hw = (w - 1) / 2.

    I, J = np.meshgrid(np.linspace(-hh, hh, x.shape[-2]), np.linspace(-hw, hw, x.shape[-1]))
    C = np.r_[[I, J]]
    D4C = np.einsum('...ij,jkl->...ikl', D4, C)
    D4C[:, 0] += hh
    D4C[:, 1] += hw
    D4C = D4C.astype(int)

    x_out = np.empty_like(x)
    for i in range(x.shape[0]):
        I, J = D4C[g[i]]
        x_out[i, :] = x[i][:, J, I]

    return x_out


def flip_transform_batch(x):

    g = np.random.randint(low=0, high=3, size=x.shape[0])

    x_out = np.empty_like(x)
    for i in range(x.shape[0]):
        if g[i] == 0:
            x_out[i] = x[i]
        elif g[i] == 1:
            x_out[i] = x[i, :, ::-1, :]
        else:
            x_out[i] = x[i, :, :, ::-1]

    return x_out


def hflip_transform_batch(x):

    g = np.random.randint(low=0, high=2, size=x.shape[0])

    x_out = np.empty_like(x)
    for i in range(x.shape[0]):
        if g[i] == 0:
            x_out[i] = x[i]
        elif g[i] == 1:
            x_out[i] = x[i, :, :, ::-1]

    return x_out


def translate_transform_batch(x):

    t = (np.random.rand(x.shape[0], 2) - 0.5) * 4

    x_out = np.empty_like(x)
    for i in range(x.shape[0]):

        # Super slow but whatever...
        shift(x[i, 0], shift=t[i], output=x_out[i, 0], order=3, mode='reflect')
        shift(x[i, 1], shift=t[i], output=x_out[i, 1], order=3, mode='reflect')
        shift(x[i, 2], shift=t[i], output=x_out[i, 2], order=3, mode='reflect')

    return x_out