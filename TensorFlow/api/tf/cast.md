# tf.cast

Last updated: 2022-09-26, 13:32
****

## 简介

```python
tf.cast(
    x, dtype, name=None
)
```

将张量强制转换为新类型。

该操作将 `x` (对 `Tensor`) 或 `x.values` (对 `SparseTensor` 或 `IndexedSlices`) 的类型转换为 `dtype`。

例如：

```python
>>> x = tf.constant([1.8, 2.2], dtype=tf.float32)
>>> tf.cast(x, tf.int32)
<tf.Tensor: shape=(2,), dtype=int32, numpy=array([1, 2])>
```

`tf.cast` 有个别名 `tf.dtypes.cast`：

```python
>>> x = tf.constant([1.8, 2.2], dtype=tf.float32)
>>> tf.dtypes.cast(x, tf.int32)
<tf.Tensor: shape=(2,), dtype=int32, numpy=array([1, 2])>
```

该操作支持的数据类型（对 `x` 和 `dtype`）包括：`uint8`, `uint16`, `uint32`, `uint64`, `int8`, `int16`, `int32`, `int64`, `float16`, `float32`, `float64`, `complex64`, `complex128`, `bfloat16`。

- 将复数类型（`complex64`, `complex128`） 转换为实数类型，只返回 `x` 的实数部分。
- 将实数类型转换为复数类型，返回的实数虚部设置为 `0`。
- 将 nan 和 inf 值转换为整数类型的行为未定义。

## 参数

|参数|说明|
|---|---|
|x|数值类型的 `Tensor`, `SparseTensor`, `IndexedSlices`。支持 uint8, uint16, uint32, uint64, int8, int16, int32, int64, float16, float32, float64, complex64, complex128, bfloat16|
|dtype|目标类型，支持类型同 `x`|
|name|操作名称，可选|

## 参考

- https://tensorflow.google.cn/api_docs/python/tf/cast
