# 训练循环基础

- [训练循环基础](#训练循环基础)
  - [简介](#简介)
  - [解决机器学习问题](#解决机器学习问题)
  - [数据](#数据)
  - [参考](#参考)

2021-12-27, 17:51
@author Jiawei Mao
***

## 简介

在前面我们已经了解了张量、变量、梯度 tape 和模块。现在，我们要把这些东西组合起来训练模型。

TensorFlow 还包括高级 API tf.keras，对基本类进行了抽象，减少了模板代码。不过下面我们只使用基础类构建训练过程。

配置：

```python
import tensorflow as tf

import matplotlib.pyplot as plt

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
```

## 解决机器学习问题

解决一个机器学习问题，通常包含如下步骤：

- 获取训练数据
- 定义模型
- 定义损失函数
- 运行训练数据，计算损失值
- 计算针对损失的梯度，并使用优化器来调整变量值以适应数据
- 评估结果

下面用一个简单的线性模型 $f(x)=x*W+b$ 解释整个训练过程，该模型包含两个变量：权重 $W$ 和偏置 $b$。

可以说这是最基本的机器学习问题：给定 $x$ 和 $y$，通过线性回归找到斜率和截距。

## 数据

监督学习需要输入（通常以 `x` 表示）和输出（表示为 `y`，也称为标签）。目标是从成对的输入和输出中学习，以便能够根据输入预测输出。

在 TensorFlow 中，输入数据以 tensor 表示，且一般为向量。包含输出的监督学习中，输出类型也是 tensor。

下面生成一批数据，在线性数据点上添加高斯噪音：

```python
# The actual line
TRUE_W = 3.0
TRUE_B = 2.0

NUM_EXAMPLES = 201

# A vector of random x values
x = tf.linspace(-2,2, NUM_EXAMPLES)
x = tf.cast(x, tf.float32)

def f(x):
  return x * TRUE_W + TRUE_B

# Generate some noise
noise = tf.random.normal(shape=[NUM_EXAMPLES])

# Calculate y
y = f(x) + noise
```

## 参考

- https://www.tensorflow.org/guide/basic_training_loops
