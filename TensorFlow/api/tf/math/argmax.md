# tf.math.argmax

Last updated: 2022-09-26, 14:37
****

## 简介

```python
tf.math.argmax(
    input,
    axis=None,
    output_type=tf.dtypes.int64,
    name=None
)
```

返回张量指定维度上最大值的索引。

对相等值，则返回最小的索引。

例如：

```python
>>> A = tf.constant([2, 20, 30, 3, 6])
>>> tf.math.argmax(A)  # A[2] 是张量 A 的最大值
<tf.Tensor: shape=(), dtype=int64, numpy=2>
>>> B = tf.constant([[2, 20, 30, 3, 6],
...                  [3, 11, 16, 1, 8],
...                  [14, 45, 23, 5, 27]])
>>> tf.math.argmax(B, 0) # 0 维最大值，每列的最大值索引
<tf.Tensor: shape=(5,), dtype=int64, numpy=array([2, 2, 0, 2, 2], dtype=int64)>
>>> tf.math.argmax(B, 1) # 1 维最大值，每行的最大值索引
<tf.Tensor: shape=(3,), dtype=int64, numpy=array([2, 2, 1], dtype=int64)>
>>> C = tf.constant([0, 0, 0, 0])
>>> tf.math.argmax(C) # 对相同值，返回最小的索引
<tf.Tensor: shape=(), dtype=int64, numpy=0>
```

## 参数

|参数|说明|
|---|---|
|input|张量|
|axis|整数，要计算的维度，默认 0|
|output_type|（可选）输出 dtype (`tf.int32` or `tf.int64`)，默认 `tf.int64`|
|name|(可选)操作名称|

返回 `output_type` 类型的 `Tensor`。

## 参考

- https://tensorflow.google.cn/api_docs/python/tf/math/argmax
