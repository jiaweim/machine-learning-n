# tf.keras.metrics.binary_crossentropy

2022-03-10, 01:00
****

## 简介

```python
tf.keras.metrics.binary_crossentropy(
    y_true, y_pred, from_logits=False, label_smoothing=0.0, axis=-1
)
```

计算二元交叉熵损失。

例如：

```python
>>> y_true = [[0, 1], [0, 0]]
>>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
>>> loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
>>> assert loss.shape == (2,)
>>> loss.numpy()
array([0.9162905 , 0.71355796], dtype=float32)
```

|参数|说明|
|---|---|
|y_true|真实值。shape = `[batch_size, d0, .. dN]`|
|y_pred|与测试。shape = `[batch_size, d0, .. dN]`|
|from_logits|`y_pred` 是否为 logit 张量。默认 `y_pred` 为概率分布|
|label_smoothing|[0, 1] 之间的 Float.如果 > 0，则通过将标签向 0.5 压缩来平滑标签，即对 target 类使用 `1. - 0.5 * label_smoothing`，对 non-target 类使用 `0.5 * label_smoothing`|
|axis|计算均值的 axis。默认 -1|

返回二元交叉熵损失值。shape = `[batch_size, d0, .. dN-1]`。

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/metrics/binary_crossentropy
