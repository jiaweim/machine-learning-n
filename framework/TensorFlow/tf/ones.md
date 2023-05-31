# tf.ones

Last updated: 2022-10-28, 10:53
****

## 简介

```python
tf.ones(
    shape,
    dtype=tf.dtypes.float32,
    name=None
)
```

创建一个所有元素都是 1 的张量。

> **aliases:** `tf.compat.v1.ones`

`tf.ones` 返回一个维度为 `shape` 的 `dtype` 类型张量，所有元素为 1.

例如：

```python
>>> tf.ones([3, 4], tf.int32) # 创建 3 行 4 列的全 1 张量
<tf.Tensor: shape=(3, 4), dtype=int32, numpy=
array([[1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1]])>
```

## 参数

|参数|说明|
|---|---|
|`shape`|integer 的 `list` 或 `tuple`，或 `int32` 类型的 1D `Tensor`|
|`dtype`|（可选）用于指定类型。默认 `tf.float32`|
|`name`|（可选）string，操作名称|

## 参考

- https://www.tensorflow.org/api_docs/python/tf/ones
