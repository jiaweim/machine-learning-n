# tf.math.add

Last updated: 2022-09-26, 17:14
****

## 简介

```python
tf.math.add(
    x, y, name=None
)
```

返回 `x + y` 元素加。

例如，list 和标量相加：

```python
>>> x = [1, 2, 3, 4, 5]
>>> y = 1
>>> tf.add(x, y)
<tf.Tensor: shape=(5,), dtype=int32, numpy=array([2, 3, 4, 5, 6])>
```

对该操作，可以直接用 `+` 运算符：

```python
>>> x = tf.convert_to_tensor([1, 2, 3, 4, 5])
>>> y = tf.convert_to_tensor(1)
>>> x + y
<tf.Tensor: shape=(5,), dtype=int32, numpy=array([2, 3, 4, 5, 6])>
```

- 张量和相同 shape 的 list 相加

```python
>>> x = [1, 2, 3, 4, 5]
>>> y = tf.constant([1, 2, 3, 4, 5])
>>> tf.add(x, y)
<tf.Tensor: shape=(5,), dtype=int32, numpy=array([ 2,  4,  6,  8, 10])>
```

> **WARNING:** 输入 `x` 或 `y` 只有一个为张量，非张量输入将采用张量输入的数据类型，这可能会导致不必要的溢出（overflow）。

例如：

```python
>>> x = tf.constant([1, 2], dtype=tf.int8)
>>> y = [2 ** 7 + 1, 2 ** 7 + 2]
>>> tf.add(x, y) # 数据太大，超出 int8 范围，溢出
<tf.Tensor: shape=(2,), dtype=int8, numpy=array([-126, -124], dtype=int8)>
```

- 两个不同 shape 的输入相加，广播规则同 NumPy。

两个输入数组的 shape 逐元素进行比较，从最后一个维度开始，要么维度相同，要么其中一个为 1。例如：

```python
>>> x = np.ones(6).reshape(1, 2, 1, 3)
>>> y = np.ones(6).reshape(2, 1, 3, 1)
>>> tf.add(x, y).shape.as_list() # 每个维度，大小为 1 的维度被广播
[2, 2, 3, 3]
```

再比如：

```python
>>> x = np.ones([1, 2, 1, 4])
>>> y = np.ones([3, 4])
>>> tf.add(x, y).shape.as_list() # 从最后一个维度开始计算
[1, 2, 3, 4]
```

该逐元素操作的降维版本为 [tf.math.reduce_sum](https://tensorflow.google.cn/api_docs/python/tf/math/reduce_sum)。

## 参数

|参数|说明|
|---|---|
|x|`tf.Tensor`，支持类型: bfloat16, half, float32, float64, uint8, int8, int16, int32, int64, complex64, complex128, string|
|y|`tf.Tensor`，和 `x` 类型必须相同|
|name|（可选）操作名称|

## 参考

- https://tensorflow.google.cn/api_docs/python/tf/math/add
