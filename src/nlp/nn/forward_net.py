import numpy as np


class Sigmoid:
    def __init__(self):
        self.params = []  # sigmoid 没有要学习的参数

    def forward(self, x):
        return 1 / (1 + np.exp(-x))


class Affine:
    """
    全连接层实现
    """

    def __init__(self, W, b):
        self.params = [W, b]  # 初始化权重和偏置

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        return out


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # 初始化权重和偏置
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        # 生成层
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        # 整理参数
        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


x = np.random.randn(10, 2)
model = TwoLayerNet(2, 4, 3)
s = model.predict(x)  # 输出 shape 10x3
print(s)
