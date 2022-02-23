import numpy as np
from src.nlp.common import MatMul

# 样本的上下文数据
c0 = np.array([[1, 0, 0, 0, 0, 0, 0]])  # 1x7
c1 = np.array([[0, 0, 1, 0, 0, 0, 0]])

# 权重的初始值
W_in = np.random.randn(7, 3)  # 7x3
W_out = np.random.randn(3, 7)

# 生成层
in_layer0 = MatMul(W_in)  # 1x3
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)

# 正向传播
h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
h = 0.5 * (h0 + h1)
s = out_layer.forward(h)  # 输出属于各个 word 的打分，加一个 softmax，就是概率啦

print(s)
