# tf.fill

Last updated: 2022-10-28, 12:26
****

## 简介

```python
tf.fill(
    dims, value, name=None
)
```

> **aliases:** `tf.compat.v1.fill`

创建一个用指定标量填充的张量。

`tf.fill` 创建一个 shapee 为 `dims` 的张量，并用 `value` 填充。

例如：

```python
>>> tf.fill([2, 3], 9)
<tf.Tensor: shape=(2, 3), dtype=int32, numpy=
array([[9, 9, 9],
       [9, 9, 9]])>
```

`tf.fill` 在 graph 运行时计算，并基于其它运行时 `tf.Tensor` 支持动态 shape，不像 `tf.constant(value, shape=dims)` 将值嵌入为 `Const` node。

## 参数

|参数|说明|
|---|---|
|dims|非负 1-D 序列。指定输出 `tf.Tensor` 的 shape。序列类型 `int32`, `int64`|
|value|填充值|
|name|（可选）返回的 `tf.Tensor` 名称|

**Returns**

shape 为 `dims` 类型与 `value` 相同的 tf.Tensor。

## 参考

- https://tensorflow.google.cn/api_docs/python/tf/fill
