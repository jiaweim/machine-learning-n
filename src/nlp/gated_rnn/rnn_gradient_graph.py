import numpy as np
import matplotlib.pyplot as plt

N = 2  # mini-batch
H = 3  # 隐藏状态向量的维数
T = 20  # 时序数据的长度

dh = np.ones((N, H))  # 将梯度初始设置为 1，后面根据反向传播 MatMul 次数反复更新 dh

np.random.seed(3)

Wh = np.random.randn(H, H)

norm_list = []
for t in range(T):
    dh = np.dot(dh, Wh.T)
    norm = np.sqrt(np.sum(dh ** 2)) / N
    norm_list.append(norm)

print(norm_list)
# 绘制图形
plt.plot(np.arange(len(norm_list)), norm_list)
plt.xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
plt.xlabel('time step')
plt.ylabel('norm')
plt.show()
