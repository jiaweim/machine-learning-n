# 训练检查点

2022-03-18, 01:12

## 简介

保存 TensorFlow 模型有两种含义：

- 检查点（Checkpoints）
- SavedModel

检查点（checkpoint）捕获模型使用的所有参数值（`tf.Variable` 对象）。检查点不包含模型定义的任何计算，因此只有在保存参数的源代码可用时才有用。

SavedModel 格式除了参数值（checkpoint），还包含模型定义的计算。这种格式的模型不依赖于创建模型的源代码。因此，可用于通过 TensorFlow Serving, TensorFlow Lite, TensorFlow.js 或其它编程语言的程序部署。

本指南介绍检查点的读写 API。

## 配置

```python
import tensorflow as tf
```

```python
class Net(tf.keras.Model):
    """一个简单的线性模型"""

    def __init__(self):
        super(Net, self).__init__()
        self.l1 = tf.keras.layers.Dense(5)

    def call(self, x):
        return self.l1(x)
```

```python
net = Net()
```

## 从 tf.keras 训练 API 保存



## 参考

- https://www.tensorflow.org/guide/checkpoint
