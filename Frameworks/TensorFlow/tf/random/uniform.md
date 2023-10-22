# uniform

Last updated: 2022-08-04, 11:15
@author Jiawei Mao
****

## 简介

```python
tf.random.uniform(
    shape,
    minval=0,
    maxval=None,
    dtype=tf.dtypes.float32,
    seed=None,
    name=None
)
```

从均匀分布生成随机数。

生成 `[minval, maxval)` 范围内的满足均匀分布的随机数，前开后闭区间。

- 对浮点数，默认范围是 `[0, 1)`
- 对整数，至少要指定 `maxval`。

对整数，随机整数会有轻微的 bias，除非 `maxval - minval` 刚好是 2 的指数幂。当 `maxval - minval` 显著小于输出范围时，如 `2**32` 或 `2**64`，bias 很小。

**返回**：指定 shape 的 tensor，填充随机均值分布值。

当 `dtype` 为整数且未指定 `maxval` 时，抛出 `ValueError`。

## 参数

|参数|说明|
|---|---|
|shape|1-D 整数 tensor 或 Python 数组。指定生成张量的 shape|
|minval|随机范围的下限 (inclusive)，默认 0。`dtype` 类型的 tensor 或 Python 值，广播为 `shape`(整数不支持广播，因此必须为标量)|
|maxval|随机范围上限 (exclusive)，如果 `dtype` 为浮点数，默认为 1.|
|dtype|类型: float16, bfloat16, float32, float64, int32, or int64. 默认 float32.|
|seed|Python 整数。与 `tf.random.set_seed` 结合使用，可以在多次调用中创建可重复序列|
|name|操作名称 (optional)|

## 示例

```python
>>> tf.random.uniform(shape=[2]) # 默认类型 float32，区间 [0,1)
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.5281, 0.3847], dtype=float32)>

>>> tf.random.uniform(shape=[], minval=-1., maxval=0.) # shape 为空时，生成一个数
<tf.Tensor: shape=(), dtype=float32, numpy=-0.6370691>

>>> tf.random.uniform(shape=[], minval=5, maxval=10, dtype=tf.int64) # 指定类型
<tf.Tensor: shape=(), dtype=int64, numpy=7>
```

设置 `seed` 参数，可以在多次调用生成相同的随机数序列。使用 `tf.random.set_seed` 设置全局 seed 可以获得相同随机序列：

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
