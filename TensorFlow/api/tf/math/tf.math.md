# Module: tf.math

2021-06-03, 20:11
***

## 简介

数学操作。

> [!NOTE]
> 以张量为参数的函数也支持 `tf.convert_to_tensor` 接受的任何参数类型。
> TF 中逐元素二元操作遵循 numpy 风格的广播。

TensorFlow 提供了多种数学函数，包括：

- 基本算术运算和三角函数；
- 特殊的数学函数，如 `tf.math.igamma` 和 `tf.math.zeta`；
- 复数函数，如 `tf.math.imag` 和 `tf.math.angle`；
- 缩小和扫描，如 `tf.math.reduce_mean` 和 `tf.math.cumsum`；
- segment 函数，如 `tf.math.segment_sum`

## 分段函数

TF 提供了几个可用于张量 segment 进行数学计算的操作。这里，segment 是沿着张量第一维的划分，即，它定义了从第一维到 `segment_ids` 的映射。`segment_ids` 张量的 size 

例如:

```python
c = tf.constant([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]])
tf.math.segment_sum(c, tf.constant([0, 0, 1]))
```

```txt
<tf.Tensor: shape=(2, 4), dtype=int32, numpy=
array([[0, 0, 0, 0],
       [5, 6, 7, 8]])>
```

标准的 `segment_*` 函数假设 segment 索引已排序。如果索引未排序，则应使用对应的 `unsorted_segment_*` 函数。这些函数包含一个额外参数 `num_segments`，以便

## igamma

计算低正则化不完全 Gamma 函数 `P(a, x)`。

```py
tf.math.igamma(
    a, x, name=None
)
```

## imag

计算复数 tensor 的虚部。

```py
tf.math.imag(
    input, name=None
)
```

## log

计算自然对数，element-wise。

```python
tf.math.log(
    x, name=None
)
```

例如：

```python
    x = tf.constant([0, 0.5, 1, 5])
    y = tf.math.log(x)
    # tf.Tensor([      -inf -0.6931472  0.         1.609438 ], shape=(4,), dtype=float32)
```

即对每个数值挨个计算自然对数。

| **参数** | **说明** |
| --- | --- |
| x | Tensor，类型限制：bfloat16, half, float32, float64, complex64, complex128 |
| name  | 操作名称 |

## reduce_sum

计算张量元素之和。

```python
tf.math.reduce_sum(
    input_tensor, axis=None, keepdims=False, name=None
)
```

沿着 `axis` 指定的轴减少输入张量 `input_tensor` 的维数。除非 `keepdims` 为 true，否则 `axis` 上的每个条目的 rank 值都减一。如果 `keepdims` 为 true，则缩小的维度保留，长度为 1.

如果 `axis=None` ，则所有维度都降维，返回只包含一个元素的 tensor。
例如：

```python
# x 的 shape 为 （2, 3）
x = tf.constant([[1, 1, 1], [1, 1, 1]])

# 不指定 axis，计算所有元素加和
v = tf.reduce_sum(x).numpy()
assert v == 6

# 指定 axis=0，[1, 1, 1] + [1, 1, 1] = [2, 2, 2]
v = tf.reduce_sum(x, 0).numpy()
assert np.array_equal(v, np.array([2, 2, 2]))

# 指定 axis=1, [1, 1] + [1, 1] + [1, 1] = [3, 3]
v = tf.reduce_sum(x, 1).numpy()
assert np.array_equal(v, np.array([3, 3]))

# 保留维度
v = tf.reduce_sum(x, 1, keepdims=True).numpy()
assert np.array_equal(v, np.array([[3], [3]]))

# 从两个维度同时 reduce
# [1,1,1]+[1,1,1]=[2,2,2]
# 2+2+2=6
v = tf.reduce_sum(x, [0, 1]).numpy()
assert v == 6
```

| 参数 | 说明 |
| --- | --- |
| input_tensor | 输入 tensor，必须为数值类型 |
| axis | 计算的维度。如果为 `None` ，则计算所有维度，范围 [-rank(input_tensor),rank(input_tensor)] |
| keepdims | 如果为 true，则计算的维度保留，长度为 1 |

等价于 `np.sum` ，除了 numpy 会将 `uint8` , `int32` 向上转换为 `int64` ，而 tensorflow 返回类型和输入相同类型。

## zeta

计算 Hurwitz zeta 函数 $\zeta(x,q)$。

## 参考

- https://www.tensorflow.org/api_docs/python/tf/math
