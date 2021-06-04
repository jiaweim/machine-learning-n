# tf.random

- [tf.random](#tfrandom)
  - [uniform](#uniform)

2021-06-03, 20:10
***

## uniform

以均匀分布生成随机值。

```py
tf.random.uniform(
    shape, minval=0, maxval=None, dtype=tf.dtypes.float32, seed=None, name=None
)
```

生成在 `[minval, maxval)` 范围内满足均匀分布的值。

对浮点类型，默认范围为 `[0, 1)` 。

对整数，至少要要指定 `maxval` 参数。并且除非 `maxval - minval` 是 2 的幂，否则随机整数会略有偏差。

| **参数** | **说明** |
| --- | --- |
| shape | 输出 tensor 的shape，1 D 整数 Tensor 或 Python 数组 |
| minval | `dtype` 类型的 Tensor 或  |
| name | 可选参数，生成 tensor 的名称 |



