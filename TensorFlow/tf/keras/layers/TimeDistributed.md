# tf.keras.layers.TimeDistributed

2022-01-30, 23:32

## 简介

```python
tf.keras.layers.TimeDistributed(
    layer, **kwargs
)
```

该包装器将一个 layer 应用到输入的每个时间步。

输入数据至少 3D，维度 #1 作为时间维度处理。

以 batch_size=32 的视频样本为例，每个样本是 128x128 RGB 图像，`channels_last` 格式，包含 10 个时间步，即输入 shape 为 `(32, 10, 128, 128, 3)`。

此时可以用 `TimeDistributed` 对 10 个时间步各自应用相同的 `Conv2D`：

```python
>>> inputs = tf.keras.Input(shape=(10, 128, 128, 3))
>>> conv_2d_layer = tf.keras.layers.Conv2D(64, (3, 3))
>>> outputs = tf.keras.layers.TimeDistributed(conv_2d_layer)(inputs)
>>> outputs.shape
TensorShape([None, 10, 126, 126, 64])
```

因为 `TimeDistributed` 对每个时间步应用相同的 `Conv2D` 实例，所以对每个时间步使用了相同的权重。

## 调用参数

- **inputs**

shape 为 `(batch, time, ...)` 的输入张量，或所有张量 shape 都为 `(batch, time, ...)` 的嵌套张量。

- **training**



## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/layers/TimeDistributed
