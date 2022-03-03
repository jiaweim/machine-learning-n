# tf.nn.softmax

2022-03-03, 16:08
***

## 简介

```python
tf.nn.softmax(
    logits, axis=None, name=None
)
```

计算 softmax 激活值，返回的 `Tensor` 与 `logits` 具有相同的类型和 shape。

用在多类别预测。softmax 输出加和为 1.

该函数的功能等价于：

```python
softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)
```

## 参数

|参数|说明|
|---|---|
|logits|A non-empty Tensor. Must be one of the following types: half, float32, float64.|
|axis|The dimension softmax would be performed on. The default is -1 which indicates the last dimension.|
|name|A name for the operation (optional).|

## 示例

```python
>>> softmax = tf.nn.softmax([-1, 0., 1.])
>>> softmax
<tf.Tensor: shape=(3,), dtype=float32, numpy=array([0.09003057, 0.24472848, 0.6652409 ], dtype=float32)>
>>> sum(softmax)
<tf.Tensor: shape=(), dtype=float32, numpy=0.99999994>
```

## 参考

- https://www.tensorflow.org/api_docs/python/tf/nn/softmax
