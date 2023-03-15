# tf.math.l2_normalize

## 简介

```python
tf.math.l2_normalize(
    x, axis=None, epsilon=1e-12, name=None, dim=None
)
```

沿指定维度 `axis` 使用 L2 范数归一化。

- 对 1D 张量且 `axis = 0`，计算：

```python
output = x / sqrt(max(sum(x**2), epsilon))
```

对高维 `x`，则沿 `axis` 单独归一化每个 1D 切片。

1D 张量示例：

```python
>>> x = tf.constant([3.0, 4.0])
>>> tf.math.l2_normalize(x).numpy()
array([0.6, 0.8], dtype=float32)
```

2D 张量示例：

```python
>>> x = tf.constant([[3.0], [4.0]])
>>> tf.math.l2_normalize(x, 0).numpy()
array([[0.6],
     [0.8]], dtype=float32)
```

```python
>>> x = tf.constant([[3.0], [4.0]])
>>> tf.math.l2_normalize(x, 1).numpy()
array([[1.0],
     [1.0]], dtype=float32)
```

**参数：**

- **x**	- `Tensor`
- **axis**	- 要归一化的维度。整数标量或向量。
- **epsilon** - 归一化的最小值。如果 norm < sqrt(epsilon)，则使用 `sqrt(epsilon)` 作为分母。
- **name** - (optional) 该操作名称。

## 参考

- https://www.tensorflow.org/api_docs/python/tf/math/l2_normalize
