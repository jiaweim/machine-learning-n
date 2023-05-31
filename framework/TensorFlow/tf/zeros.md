# tf.zeros

Last updated: 2022-10-28, 11:22
****

## 简介

```python
tf.zeros(
    shape,
    dtype=tf.dtypes.float32,
    name=None
)
```

创建一个所有元素都是 0 的张量。

> **aliases:** `tf.compat.v1.zeros`

`tf.zeros` 返回一个维度为 `shape` 的 `dtype` 类型张量，所有元素为 0.

例如：

```python
>>> tf.zeros([3, 4], tf.int32) # 创建 3 行 4 列的全 0 张量
<tf.Tensor: shape=(3, 4), dtype=int32, numpy=
array([[0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]])>
```

## 参数

|参数|说明|
|---|---|
|`shape`|integer 的 `list` 或 `tuple`，或 `int32` 类型的 1D `Tensor`|
|`dtype`|（可选）用于指定类型。默认 `tf.float32`|
|`name`|（可选）string，操作名称|

**Returns**

所有元素为 0 的 `Tensor`。

## 参考

- https://tensorflow.google.cn/api_docs/python/tf/zeros
