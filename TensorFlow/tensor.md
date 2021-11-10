# Tensor

- [Tensor](#tensor)
  - [简介](#简介)
  - [创建 Tensor](#创建-tensor)
    - [tf.constant](#tfconstant)
    - [tf.fill](#tffill)
    - [tf.zeros](#tfzeros)
    - [tf.zeros_like](#tfzeros_like)
    - [tf.ones](#tfones)
    - [tf.linspace](#tflinspace)
    - [tf.range](#tfrange)

2021-06-03, 20:14
***

## 简介

张量（tensor）是单一类型（ `dtype` ）的多维数组。与 Python 字符串一样，所有 tensor 都是 immutable，如果要修改 tensor，只能创建一个新的 tensor。

TensorFlow 支持三种类型的 Tensor:

1. Constants，常量 tensor，其数值不允许改变；
2. Variables，变量 tensor；
3. Placeholders，用于将数值馈送到 tensorflow graph.

TensorFlow 支持的 tensor 类型在 `tf.dtypes.DType` 中有详细列表。



## 创建 Tensor

### tf.constant

创建一个 tensor。

```python
tf.constant(
    value, dtype=None, shape=None, name='Const'
)
```

> **NOTE:**这个函数和 `tf.convert_to_tensor` 没有本质区别。之所以称为 `tf.constant` 是因为其中的 `value` 在 `tf.Graph` 中 `Const` 模式嵌入。

- 如果没有指定 `dtype` ，则通过 `value` 的类型推测。例如：

```python
# 从 Python 列表创建一维 Tensor
a = tf.constant([1, 2, 3, 4, 5, 6])
assert a.dtype == tf.int32
# 从 NumPy 创建
a = tf.constant(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64))
assert a.dtype == tf.int64
```

### tf.fill

创建一个用指定标量填充的张量。

```python
tf.fill(
    dims, value, name=None
)
```

创建一个用 `value` 填充维度为 `dims` 的张量。
例如：

```python
    t = tf.fill([2, 3], 9) # 2行3列，全部为 9
    tf.assert_equal(t, tf.constant([[9, 9, 9], [9, 9, 9]]))
```

`tf.fill` 在图形运行时计算，因此支持动态 `dims` 参数，如运行时其他的 `tf.Tensor`，而不像 `tf.constant(value, shape=dims)` ，包含的值必须为常量。

| **参数** | **说明** |
| --- | --- |
| dims | 1 维非负数序列，表示生成 tensor 的 shape。类型为 int32 或 int64 |
| value | 填充的值 |
| name | 可选参数，生成 tensor 的名称 |

类似于 `np.full` 函数。不过 `numpy` 支持更多的参数。在 numpy 中可以通过指定一个数值 `np.full(5, value)` 创建 一个1-D数组，TensorFlow 不支持该语法。

### tf.zeros

创建所有值为 0 的 tensor。

```python
tf.zeros(
    shape, dtype=tf.dtypes.float32, name=None
)
```

创建一个类型为 `dtype` ，shape 为 `shape` 所有值为 0 的 tensor。例如：

```python
t = tf.zeros([3, 4], tf.int32)
tf.assert_equal(t, tf.constant([[0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0]], dtype=tf.int32))
```

### tf.zeros_like

创建一个所有值为 0 的 tensor。

```python
tf.zeros_like(
    input, dtype=None, name=None
)
```

和 `tf.zeros` 类似，只是 shape 由输入的 tensor 或数组对象提供，即常见一个类型和 shape 与 `input` 相同的值全部为 0 的 tensor。也可以通过 `dtype` 指定类型。

- 例如：根据指定 tensor 创建 0 tensor

```python
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
t = tf.zeros_like(tensor)
tf.assert_equal(t, tf.constant([[0, 0, 0], [0, 0, 0]]))
```

- 例如：指定类型

```python
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
t = tf.zeros_like(tensor, dtype=tf.float32)
tf.assert_equal(t, tf.constant([[0, 0, 0], [0, 0, 0]], tf.float32))
```

- 例如：根据指定数组创建

```python
t = tf.zeros_like([[1, 2, 3], [4, 5, 6]])
tf.assert_equal(t, tf.constant([[0, 0, 0], [0, 0, 0]]))
```

### tf.ones

创建一个所有值为 1 的 tensor。

```python
tf.ones(shape, dtype=tf.dtypes.float32, name=None)
```

### tf.linspace

沿着给定的轴生成等间隔的数值序列。

```python
tf.linspace(start, stop, num, name=None, axis=0)
```

如果 `num > 1` ，则间隔为 `stop - start / num - 1` ，随后一个值为 `stop` ；
如果 `num <= 0` ，抛出 `ValueError` 。

| **参数** | **说明** |
| --- | --- |
| start | 起始值， `Tensor` 类型，支持类型 bfloat16, float32, float64 |
| stop | 末尾值， `Tensor` 类型，必须和 `start` 具有相同 type 和 shape |
| num | 生成值的个数， `Tensor` 类型，支持 int32 和 int64 |
| name | 可选参数，操作名称 |
| axis | 沿哪个轴执行操作，在提供 N-D tensor 时使用 |

返回一个和 start 相同类型的 `Tensor` 。

- 例如：在 10 到 12 之间生成等间距的 3 个数

```python
t = tf.linspace(10.0, 12.0, 3, name="linspace")
tf.assert_equal(t, tf.constant([10., 11., 12.]))
```

- `start` 和 `stop` 可以是任意 size 的 tensor。例如沿着 axis=0 生成 tensor：

```python
t = tf.linspace([0., 5.], [10., 40.], 5, axis=0)
tf.assert_equal(t, tf.constant([[0., 5.],
                                [2.5, 13.75],
                                [5., 22.5],
                                [7.5, 31.25],
                                [10., 40.]]))
```

在 0 到 10 之间生成 5 个值，5 到 40 之间生成 5 个值。

- 沿着 axis=1 生成 tensor：

```python
t = tf.linspace([0., 5.], [10., 40.], 5, axis=-1)
tf.assert_equal(t, tf.constant([[0., 2.5, 5., 7.5, 10.],
                                [5., 13.75, 22.5, 31.25, 40.]]))
```

### tf.range

```python
tf.range(limit, delta=1, dtype=None, name='range')
tf.range(start, limit, delta=1, dtype=None, name='range')
```

以 `start` 生成数值序列，值间隔为 `delta` ，直到但不包括 `limit` 。

| **参数** | **说明** |
| --- | --- |
| start | 标量值，如果 `limit` 不为 None，则为第一个值；否则为 `limit` ，第一个值默认为 0 |
| limit | 标量值，序列上限，不包含 |
| delta | 增量，默认为 1 |
| dtype | 类型 |
| name | 操作名称，默认为 "range" |

- 例如，不包含 limit 值：

```python
t = tf.range(3, 18, 3)
tf.assert_equal(t, tf.constant([3, 6, 9, 12, 15]))
```

- delta 为负值：

```python
t = tf.range(start=3, limit=1, delta=-0.5)
tf.assert_equal(t, tf.constant([3., 2.5, 2., 1.5]))
```

- 只有一个参数，表示 limit

```python
t = tf.range(5)
tf.assert_equal(t, tf.constant([0, 1, 2, 3, 4]))
```
