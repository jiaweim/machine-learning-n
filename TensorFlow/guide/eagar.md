# eager 执行

- [eager 执行](#eager-执行)
  - [简介](#简介)
  - [设置和基本用法](#设置和基本用法)
  - [动态控制流](#动态控制流)
  - [参考](#参考)

2021-06-03, 20:13
***

## 简介

TensorFlow 1.x 定义静态计算图，这种描述式编程让很多人不习惯。Pytorch，另一个受欢迎的深度学习包，使用命令式编程和动态的方式定义计算图，可以随时定义、修改以及执行 graph node，不需要特殊的 session 接口或 placeholder。这就是所谓的**即时执行**（eager execution），此时模型定义是动态的，并且立即执行。

TensorFlow 2.x 支持 eager 执行。因此不再需要先定义静态计算图再执行。所有模型都可以动态定义并立即执行。所有的 `tf.keras` API 都和 eager 执行兼容。

TensorFlow 的 eager 执行是一个命令式的编程环境，不需要构建 graph 就可以直接计算操作结果。Eager 执行使得 TensorFlow 的学习以及模型调试更为容易，也减少了样板代码。

Eager 执行是一个用于研究和实验的机器学习平台，提供了如下工具：

- 直观的界面，自然地使用 Python 数据结构构造代码，快速迭代小型模型和数据；
- 简化调试，可以直接调用 ops 检查正在运行的模型并测试更改。使用标准 Python 调试工具进行即时错误报告；
- 自然控制流，使用 Python 控制流代替图形控制流，从而简化了动态模型的构建。

## 设置和基本用法

```py
import os
import tensorflow as tf
import cProfile
```

在 Tensorflow 2.0 中 Eager 执行默认启用：

```py
>>> tf.executing_eagerly()
True
```

所以执行的 TensorFlow 操作，都立刻返回结果：

```python
x = [[2.]]
m = tf.matmul(x, x)
print(f'hello, {m}')

# hello, [[4.]]
```

Eager 执行使得 TensorFlow 立刻求值并返回结果，此时 `tf.Tensor` 对象引用的是具体值，而不是计算网络图中节点的句柄。由于没有在会话（session）中构建和运行计算图，因此更容易使用 `print()` 或 debugger 检查调试。评估、打印以及检查 tensor 值都不会中断梯度的计算。

Eager 执行和 NumPy 配合使用十分合适。 `tf.math` 操作将 Python 对象和 NumPy 数组转换为 `tf.Tensor` 对象， `tf.Tensor.numpy` 方法返回响应的 NumPy `ndarray` 对象。

```python
a = tf.constant([[1, 2], [3, 4]])
print(a)

# tf.Tensor(
# [[1 2]
#  [3 4]], shape=(2, 2), dtype=int32)
```

支持广播：

```python
b = tf.add(a, 1)
print(b)

# tf.Tensor(
# [[2 3]
#  [4 5]], shape=(2, 2), dtype=int32)
```

支持运算符重载：

```python
print(a*b)

# tf.Tensor(
# [[ 2  6]
#  [12 20]], shape=(2, 2), dtype=int32)
```

NumPy 操作：

```python
import numpy as np

c = np.multiply(a, b)
print(c)

# [[ 2  6]
#  [12 20]]
```

从 tensor 获得 numpy 值：

```python
print(a.numpy())
# [[1 2]
#  [3 4]]
```

## 动态控制流

Eager 执行的优点是，在执行模型时可以使用宿主编程语言的所有功能。例如，很容易写个 fizzbuzz:

```python
def fizzbuzz(max_num):
    counter = tf.constant(0)
    max_num = tf.convert_to_tensor(max_num)
    for num in range(1, max_num.numpy()+1):
        num = tf.constant(num)
        if int(num % 3) == 0 and int(num % 5) == 0:
            print('FizzBuzz')
        elif int(num % 3) == 0:
            print('Fizz')
        elif int(num % 5) == 0:
            print('Buzz')
        else:
            print(num.numpy())
        counter += 1
fizzbuzz(15)

# 1
# 2
# Fizz
# 4
# Buzz
# Fizz
# 7
# 8
# Fizz
# Buzz
# 11
# Fizz
# 13
# 14
# FizzBuzz
```

## 参考

- https://tensorflow.google.cn/guide/eager
- https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/eager.ipynb
