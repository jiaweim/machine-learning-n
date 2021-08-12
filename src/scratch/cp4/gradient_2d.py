# coding: utf-8
# cf.http://d.hatena.ne.jp/white_wheels/20100327/p3
import matplotlib.pylab as plt
import numpy as np


def _numerical_gradient_no_batch(f, x):
    """
    由全部变量的偏导数汇总而成的向量称为梯度。
    :param f: 待计算梯度的函数
    :param x: numpy 数组，对该数组的各个元素求数值微分
    """
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # 生成和 x 形状相同的零数组

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值

    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)

        return grad


def function_2(x):
    """
    对两个变量，计算偏导数
    """
    if x.ndim == 1:
        return np.sum(x ** 2)
    else:
        return np.sum(x ** 2, axis=1)


def tangent_line(f, x):
    d = numerical_gradient(f, x)
    print(d)
    y = f(x) - d * x
    return lambda t: d * t + y


if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)

    X = X.flatten()
    Y = Y.flatten()

    grad = numerical_gradient(function_2, np.array([X, Y]))

    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666")  # ,headwidth=10,scale=40,color="#444444")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()
