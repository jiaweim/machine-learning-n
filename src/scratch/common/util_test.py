import numpy as np
from .util import im2col


def test_im2col():
    x1 = np.random.rand(1, 3, 7, 7)
    col1 = im2col(x1, 5, 5, stride=1, pad=0)
    assert col1.shape == (9, 75)
