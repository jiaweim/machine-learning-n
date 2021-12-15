# metrics

- [metrics](#metrics)
  - [简介](#简介)
  - [binary_crossentropy](#binary_crossentropy)
  - [参考](#参考)

2021-12-14, 15:15
***

## 简介

指标（Metric）是用来判断模型性能的函数。

指标函数与损失函数相似，不同的是，在训练模型时不使用指标函数的计算结果。任何损失函数都可以作为指标函数使用。

## binary_crossentropy

计算二元交叉熵损失。

```python
tf.keras.metrics.binary_crossentropy(
    y_true, y_pred, from_logits=False, label_smoothing=0.0, axis=-1
)
```

例如：

```python
>>> y_true = [[0, 1], [0, 0]]
>>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
>>> loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
>>> assert loss.shape == (2,)
>>> loss.numpy()

```

## 参考

- https://tensorflow.google.cn/api_docs/python/tf/keras/metrics
- https://keras.io/api/metrics/
