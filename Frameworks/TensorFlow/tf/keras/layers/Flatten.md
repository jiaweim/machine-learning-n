# tf.keras.layers.Flatten

2022-03-04, 00:18
****

## 简介

```python
tf.keras.layers.Flatten(
    data_format=None, **kwargs
)
```

将输入展平，不影响批量大小。

`data_format` 参数指定输入数据维度顺序，可选项：

- `channels_last`，默认值，对应 shape `(batch, ..., channels)`
- `channels_first`，对应 shape `(batch, channels, ...)`

## 示例

```python
>>> model = tf.keras.Sequential()
>>> model.add(tf.keras.layers.Conv2D(64, 3, 3, input_shape=(3, 32, 32)))
>>> model.output_shape
(None, 1, 10, 64)
```

```python
>>> model.add(Flatten())
>>> model.output_shape
(None, 640)
```

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten
