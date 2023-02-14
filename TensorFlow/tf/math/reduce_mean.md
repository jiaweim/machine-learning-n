# tf.math.reduce_mean

2022-02-23, 16:12
****

## 简介

```python
tf.math.reduce_mean(
    input_tensor, axis=None, keepdims=False, name=None
)
```

通过计算指定 `axis` 元素的平均值减少输入 `input_tensor` 的维数。如果 `keepdims` 为 true，则保持对应维度，长度为 1；否则张量的维度对每个 `axis` 的值都减少 1.

如果 `axis` 为 None，对所有维度进行计算，返回包含一个元素的张量。

## 参数

|参数|说明|
|---|---|
|input_tensor|The tensor to reduce. Should have numeric type.|
|axis|The dimensions to reduce. If None (the default), reduces all dimensions. Must be in the range [-rank(input_tensor),rank(input_tensor)).|
|keepdims|If true, retains reduced dimensions with length 1.|
|name|A name for the operation (optional).|

## 示例

```python
>>> x = tf.constant([[1., 1.], [2., 2.]])
>>> tf.reduce_mean(x) # 默认计算所有元素均值
<tf.Tensor: shape=(), dtype=float32, numpy=1.5>
>>> tf.reduce_mean(x, 0) # 沿着第 1 个轴计算均值
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([1.5, 1.5], dtype=float32)>
>>> tf.reduce_mean(x, 1) # 沿着第 2 个轴计算均值
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([1., 2.], dtype=float32)>
```

## 参考

- https://www.tensorflow.org/api_docs/python/tf/math/reduce_mean
