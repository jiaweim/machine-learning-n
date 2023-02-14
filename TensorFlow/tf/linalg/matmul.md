# tf.linalg.matmul

Last updated: 2022-10-11, 13:08
****

## 简介

```python
tf.linalg.matmul(
    a,
    b,
    transpose_a=False,
    transpose_b=False,
    adjoint_a=False,
    adjoint_b=False,
    a_is_sparse=False,
    b_is_sparse=False,
    output_type=None,
    name=None
)
```

矩阵乘法。

输入必须是秩 $\ge 2$ 的张量，且内部的两个维度满足矩阵乘法所需维数，任何外部维数和 batch size 匹配。

两个矩阵的类型必须相同。支持类型：`bfloat16`, `float16`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.

通过将对应的 flag 设置为 `True`，两个矩阵都可以进行转置或临接（共轭+转置），默认为 `False`。

如果矩阵包含大量的零，将对应的 `a_is_sparse` 或 `b_is_sparse` 设置为 `True` 会使用一个更有效的算法，默认为 `False`。这种优化算法只适用于 `bfloat16` 和 `float32` 类型的普通矩阵（二阶矩阵）。

- 简单的二阶矩阵乘法

```python
>>> a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
>>> a  # 2-D tensor
<tf.Tensor: shape=(2, 3), dtype=int32, numpy=
array([[1, 2, 3],
       [4, 5, 6]])>
>>> b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
>>> b  # 2-D tensor
<tf.Tensor: shape=(3, 2), dtype=int32, numpy=
array([[ 7,  8],
       [ 9, 10],
       [11, 12]])>
>>> c = tf.matmul(a, b)
>>> c  # `a` * `b`
<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
array([[ 58,  64],
       [139, 154]])>
```

- batch 为 `[2]` 的批量矩阵乘法

```python
>>> a = tf.constant(np.arange(1, 13, dtype=np.int32), shape=[2, 2, 3])
>>> a  # 3-D tensor
<tf.Tensor: shape=(2, 2, 3), dtype=int32, numpy=
array([[[ 1,  2,  3],
        [ 4,  5,  6]],

       [[ 7,  8,  9],
        [10, 11, 12]]])>
>>> b = tf.constant(np.arange(13, 25, dtype=np.int32), shape=[2, 3, 2])
>>> b  # 3-D tensor
<tf.Tensor: shape=(2, 3, 2), dtype=int32, numpy=
array([[[13, 14],
        [15, 16],
        [17, 18]],

       [[19, 20],
        [21, 22],
        [23, 24]]])>
>>> c = tf.matmul(a, b)
>>> c  # `a` * `b`
<tf.Tensor: shape=(2, 2, 2), dtype=int32, numpy=
array([[[ 94, 100],
        [229, 244]],

       [[508, 532],
        [697, 730]]])>
```

从 Python >= 3.5 开始支持 `@` 运算符。所以下面两行是等价的：

```python
d = a @ b @ [[10], [11]]
d = tf.matmul(tf.matmul(a, b), [[10], [11]])
```

## 参数

|参数|说明|
|---|---|
|a|`tf.Tensor`，类型为 `float16`, `float32`, `float64`, `int32`, `complex64` 或 `complex128`，且 rank > 1|
|b|`tf.Tensor`，类型同 a|
|transpose_a|True 表示在乘之前对 a 进行转置|
|transpose_b|True 表示在乘之前对 b 进行转置|
|adjoint_a|True 表示在乘之前对 a 进行共轭和转置|
|adjoint_b|True 表示在乘之前对 b 进行共轭和转置|
|a_is_sparse|True 表示将 a 视为稀疏矩阵。注意，它不支持 `tf.sparse.SparseTensor`，只是假设 a 中大部分值为 0 来进行优化。`tf.sparse.SparseTensor` 的乘法请参考 [tf.sparse.sparse_dense_matmul](https://tensorflow.google.cn/api_docs/python/tf/sparse/sparse_dense_matmul)|
|b_is_sparse|True 表示将 b 视为稀疏矩阵。注意，它不支持 `tf.sparse.SparseTensor`，只是假设 a 中大部分值为 0 来进行优化。`tf.sparse.SparseTensor` 的乘法请参考 [tf.sparse.sparse_dense_matmul](https://tensorflow.google.cn/api_docs/python/tf/sparse/sparse_dense_matmul)|
|output_type|输出数据类型。默认 `None` 表示输出类型和输入类型相同。该参数目前只适用于输入张量类型为 (u)int8 且输出类型为 int32|
|name|（可选）操作名称|

返回类型与 `a` 和 `b` 相同的 `tf.Tensor`，其中最内部的矩阵是 `a` 和 `b` 中相应矩阵的乘积。例如，如果所有 transpose 和 adjoint 参数都是 `False`，则：

```python
output[..., i, j] = sum_k (a[..., i, k] * b[..., k, j])
```

> 注意：这是矩阵乘积，不是逐元素乘积。

## 参考

- https://www.tensorflow.org/api_docs/python/tf/linalg/matmul
- https://tensorflow.google.cn/api_docs/python/tf/linalg/matmul
