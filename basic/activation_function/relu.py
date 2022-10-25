import matplotlib.axes._axes as axes
import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np


def relu(x):
    return np.where(x < 0, 0, x)


def relu_derivative(x):
    return np.greater(x, 0).astype(int)


def sigmoid(x):
    pass


xs = np.arange(-5, 6)

fig, (ax1, ax2) = plt.subplots(2,
                               1)  # type:figure.Figure, (axes.Axes, axes.Axes)


def set_splines(ax):
    ax.spines[["left", "bottom"]].set_position(
        ("data", 0))  # 通过值来设置 x 和 y 轴位置，都设置 0 位置
    ax.spines[["top", "right"]].set_visible(False)  # 隐藏


set_splines(ax1)
ax1.plot(xs, relu(xs), linewidth=2.4)
ax1.set_title("Activation Function")

set_splines(ax2)
ax2.plot(xs, relu_derivative(xs), linewidth=2.4)
ax2.set_title("Derivative Function")

plt.show()
