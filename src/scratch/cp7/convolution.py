import numpy as np

from ..common.util import im2col


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        """
        初始化，接受滤波器的参数
        :param W: weight
        :param b: bias
        :param stride: 步幅
        :param pad: 填充
        """
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape  # 滤波器
        N, C, H, W = x.shape

        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_w = self.W.reshape(FN, -1).T  # 滤波器展开为 1 列
        out = np.dot(col, col_w) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        return out
