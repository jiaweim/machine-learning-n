# coding: utf-8
import numpy as np


def identity_function(x):
    """
    恒等函数，将输入按原样输出，一种激活函数
    """
    return x


def step_function(x):
    """
    阶跃函数，当输出超过 0 时，输出1，否则输出 0
    `dtype` 指定类型，就布尔类型转换为整数，这样 False 就变为 0，True 变为 1
    """
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x >= 0] = 1
    return grad


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
    """
    均方差损失函数
    :param y: 输出值
    :param t: 监督数据，即正确值
    :return: 损失值
    """
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    """
    交叉熵误差，实际是正确解标签输出自然数对数。
    :param y: 神经网络输出值
    :param t: 监督值
    :return: 损失值
    """
    if y.ndim == 1:  # 求单个数据的交叉熵
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    # np.arange(batch_size) 生成 0 到 batch_size - 1 的序列，
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)
