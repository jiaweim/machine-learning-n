# tf.ones

2022-02-15, 14:11
****

## 简介

创建所有元素都是 1 的张量。

```python
tf.ones(
    shape, dtype=tf.dtypes.float32, name=None
)
```

`tf.ones` 返回一个形状为 `shape` 的 `dtype` 类型张量，所有元素为 1.

例如：

```python
>>> tf.ones([3, 4], tf.int32) # 创建 3 行 4 列的全 1 张量
<tf.Tensor: shape=(3, 4), dtype=int32, numpy=
array([[1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1]])>
```

## 参数

- `shape`

A list of integers, a tuple of integers, or a 1-D Tensor of type int32.

- `dtype`

用于指定类型。Optional DType of an element in the resulting Tensor. Default is `tf.float32`.

- `name`

Optional string. A name for the operation.

## 参考

- https://www.tensorflow.org/api_docs/python/tf/ones
