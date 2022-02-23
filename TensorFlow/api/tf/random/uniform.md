# tf.random.uniform

2022-02-23, 10:35
***

## 简介

从均匀分布生成随机值。

```python
tf.random.uniform(
    shape, minval=0, maxval=None, dtype=tf.dtypes.float32, seed=None, name=None
)
```

生成的随机数遵循范围 `[minval, maxval)` 内的均匀部分，前开后闭区间。

对浮点数，默认范围是 `[0, 1)`，对整数，至少要指定 `maxval`。

在整数情况，随机整数会有轻微的 bias，除非 `maxval - minval` 刚好是 2 的指数幂。当 `maxval - minval` 显著小于输出范围时，（如 `2**32` 或 `2**64`），bias 很小。

## 参数

|参数|说明|
|---|---|
|shape|A 1-D integer Tensor or Python array. The shape of the output tensor.|
|minval|A Tensor or Python value of type dtype, broadcastable with shape (for integer types, broadcasting is not supported, so it needs to be a scalar). The lower bound on the range of random values to generate (inclusive). Defaults to 0.|
|maxval|A Tensor or Python value of type dtype, broadcastable with shape (for integer types, broadcasting is not supported, so it needs to be a scalar). The upper bound on the range of random values to generate (exclusive). Defaults to 1 if dtype is floating point.|
|dtype|The type of the output: float16, bfloat16, float32, float64, int32, or int64. Defaults to float32.|
|seed|A Python integer. Used in combination with tf.random.set_seed to create a reproducible sequence of tensors across multiple calls.|
|name|A name for the operation (optional).|

## 示例

```python
>>> tf.random.uniform(shape=[2]) # 默认类型 float32，区间 [0,1)
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.66330266, 0.6639905 ], dtype=float32)>
>>> tf.random.uniform(shape=[], minval=-1., maxval=0.) # shape 为空时，生成一个数
<tf.Tensor: shape=(), dtype=float32, numpy=-0.7678834>
>>> tf.random.uniform(shape=[], minval=5, maxval=10, dtype=tf.int64)
<tf.Tensor: shape=(), dtype=int64, numpy=7>
```

设置固定的 `seed` 参数，可以在多次调用生成相同的随机数。使用 `tf.random.set_seed` 设置全局 seed 可以获得相同随机序列：

```python
>>> tf.random.set_seed(5)
>>> tf.random.uniform(shape=[], maxval=3, dtype=tf.int32, seed=10)
<tf.Tensor: shape=(), dtype=int32, numpy=2>
>>> tf.random.uniform(shape=[], maxval=3, dtype=tf.int32, seed=10)
<tf.Tensor: shape=(), dtype=int32, numpy=0>

>>> tf.random.set_seed(5)
>>> tf.random.uniform(shape=[], maxval=3, dtype=tf.int32, seed=10)
<tf.Tensor: shape=(), dtype=int32, numpy=2>
>>> tf.random.uniform(shape=[], maxval=3, dtype=tf.int32, seed=10)
<tf.Tensor: shape=(), dtype=int32, numpy=0>
```

如果不设置 `tf.random.set_seed`，但是指定 `seed` 参数，则计算图或前面操作的更改会影响返回的值。

## 参考

- https://www.tensorflow.org/api_docs/python/tf/random/uniform
