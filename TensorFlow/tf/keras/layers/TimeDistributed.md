# tf.keras.layers.TimeDistributed

Last updated: 2023-02-15, 11:14
****

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

**参数：**

- **layer**	- `tf.keras.layers.Layer` 实例


**调用参数：**

- **inputs**

shape 为 `(batch, time, ...)` 的输入张量，或所有张量 shape 都为 `(batch, time, ...)` 的嵌套张量。

- **training**

表示该层是在训练模式还是预测模式。该参数会传递给 wrapper 的层。

- **mask**

shape 为 `(samples, timesteps)` 的张量，表示是否屏蔽指定时间步。该参数直接传递给 wrapper 层（该层需支持该 `mask` 参数）。

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/layers/TimeDistributed
