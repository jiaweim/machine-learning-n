# coding: utf-8

import numpy as np

from src.scratch.common.functions import softmax, cross_entropy_error
from src.scratch.common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.default_rng().standard_normal(size=(2, 3))  # 用高斯分布进行初始化

    def predict(self, x):
        return np.dot(x, self.W)  # 根据 weights 计算理论值，即推导值

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()
print(net.W)

p = net.predict(x)
print(p)
print(np.argmax(p))

loss = net.loss(x, t)
print(locals())

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)  # 计算在 x 值处的梯度

print(dW)
