# tf.ones_like

Last updated: 2022-10-28, 11:18

****

## 简介

```python
tf.ones_like(
    input, dtype=None, name=None
)
```

创建一个 shape 与输入 `input` 相同的全 1 张量。

给定一个张量，`tf.ones_like` 返回一个与其类型和 shape 相同，且所有元素为 1 的张量。也可以使用 `dtype` 指定返回张量的类型。

例如：

```python
>>> tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
>>> tf.ones_like(tensor)
<tf.Tensor: shape=(2, 3), dtype=int32, numpy=
array([[1, 1, 1],
       [1, 1, 1]])>
```

## 参数

|参数|说明|
|---|---|
|input|一个 `Tensor`|
|dtype|返回 Tensor 的类型，支持 `float16`, `float32`, `float64`, `int8`, `uint8`, `int16`, `uint16`, `int32`, `int64`, `complex64`, `complex128`, `bool` or `string`|
|name|（可选）操作名称|

**Returns**

所有元素为 1 的 `Tensor`。

## 参考

- https://tensorflow.google.cn/api_docs/python/tf/ones_like
