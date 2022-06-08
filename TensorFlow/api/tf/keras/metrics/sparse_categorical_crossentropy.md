# tf.keras.metrics.sparse_categorical_crossentropy

2022-03-10, 00:39
***

## 简介

```python
tf.keras.metrics.sparse_categorical_crossentropy(
    y_true, y_pred, from_logits=False, axis=-1
)
```

计算稀疏分类交叉熵损失。

```python
y_true = [1, 2]
y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
assert loss.shape == (2,)
loss.numpy()

```

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/metrics/sparse_categorical_crossentropy
