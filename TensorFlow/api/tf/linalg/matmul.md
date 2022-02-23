# tf.linalg.matmul

2022-02-23, 16:58
****

## 简介

```python
tf.linalg.matmul(
    a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False,
    a_is_sparse=False, b_is_sparse=False, output_type=None, name=None
)
```

矩阵乘法。

输入必须是秩 $\ge 2$ 的张量，且内部的两个维数满足矩阵乘法所需维数，任何外部维数和批量大小匹配。

两个矩阵类型必须相同。支持类型：bfloat16, float16, float32, float64, int32, int64, complex64, complex128.

通过将对应的标志设置为 `True`，两个矩阵都可以进行转置或临接（共轭+转置）。它们默认为 `False`。

如果矩阵包含大量的零，将对应的 `a_is_sparse` 或 `b_is_sparse` 设置为 `True` 会启用一个更有效的算法。它们默认都是 `False`。这种优化算法只适用于 bfloat16 和 float32 类型的普通矩阵（二阶矩阵）。

## 参数

|参数|说明|
|---|---|
|a|tf.Tensor of type float16, float32, float64, int32, complex64, complex128 and rank > 1.|
|b|tf.Tensor with same type and rank as a.|
|transpose_a|If True, a is transposed before multiplication.|
|transpose_b|If True, b is transposed before multiplication.|
|adjoint_a|If True, a is conjugated and transposed before multiplication.|
|adjoint_b|If True, b is conjugated and transposed before multiplication.|
|a_is_sparse|If True, a is treated as a sparse matrix. Notice, this does not support tf.sparse.SparseTensor, it just makes optimizations that assume most values in a are zero. See tf.sparse.sparse_dense_matmul for some support for tf.sparse.SparseTensor multiplication.|
|b_is_sparse|If True, b is treated as a sparse matrix. Notice, this does not support tf.sparse.SparseTensor, it just makes optimizations that assume most values in a are zero. See tf.sparse.sparse_dense_matmul for some support for tf.sparse.SparseTensor multiplication.|
|output_type|The output datatype if needed. Defaults to None in which case the output_type is the same as input type. Currently only works when input tensors are type (u)int8 and output_type can be int32.|
|name|Name for the operation (optional).|

## 示例

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

- 批量为 [2] 的批量矩阵乘法

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

## 参考

- https://www.tensorflow.org/api_docs/python/tf/linalg/matmul
