# coding: utf-8
import matplotlib.pyplot as plt

from dataset import spiral

x, t = spiral.load_data()  # x 是输入数据，t 是监督标签
print('x', x.shape)  # (300, 2)
print('t', t.shape)  # (300, 3)

# 绘制数据点
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i * N:(i + 1) * N, 0], x[i * N:(i + 1) * N, 1], s=40, marker=markers[i])
plt.show()
