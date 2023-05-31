# tf.math.sin

2022-02-23, 15:01
****

## 简介

```python
tf.math.sin(
    x, name=None
)
```

给定张量 x，该函数计算张量中每个元素的正弦值。

x 为 `Tensor` 类型，支持类型：bfloat16, half, float32, float64, complex64, complex128.

例如：

```python
import tensorflow as tf

x = tf.constant([-float("inf"), -9, -0.5, 1, 1.2, 200, 10, float("inf")])
print(tf.math.sin(x))
```

```sh
tf.Tensor(
[nan -0.41211846 -0.47942555 0.84147096 0.9320391 -0.87329733 -0.54402107 nan], shape=(8,), dtype=float32)
```

## 参考

- https://www.tensorflow.org/api_docs/python/tf/math/sin
