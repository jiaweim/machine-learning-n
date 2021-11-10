# tf.nn

- [tf.nn](#tfnn)
  - [tf.nn.softmax](#tfnnsoftmax)

2021-11-10, 13:37
***

## tf.nn.softmax

计算 softmax 激活值。

```py
tf.nn.softmax(
    logits, axis=None, name=None
)
```

用在多分类问题种，softmax 输出的所有值加和为 1.

- `logits` 为非空 `Tensor`。支持如下类型：`half`, `float32`, `float64`；
- `axis` 用于指定执行 softmax 计算的轴，默认为 -1，即最后一个维度；
- `name` 用于指定操作名称。

返回和 `logits` 等长、同类型的 `Tensor`。

该函数等价于：

```py
softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)
```

例如：

```py
>>> import tensorflow as tf
>>> softmax = tf.nn.softmax([-1, 0., 1.])
>>> softmax
<tf.Tensor: shape=(3,), dtype=float32, numpy=array([0.09003057, 0.24472848, 0.6652409 ], dtype=float32)>
>>> sum(softmax)
<tf.Tensor: shape=(), dtype=float32, numpy=0.99999994>
```

> 虽然可以将 `tf.nn.softmax` 函数作为神经网络最后一层的激活函数，这样模型的输出更容易解释，但是不推荐这么做，因为使用 softmax 输出无法对所有模型提供了一个精确且稳定的损失值计算。