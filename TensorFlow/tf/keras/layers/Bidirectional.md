# tf.keras.layers.Bidirectional

## 简介

```python
tf.keras.layers.Bidirectional(
    layer,
    merge_mode='concat',
    weights=None,
    backward_layer=None,
    **kwargs
)
```

用于构建双向 RNN。

## 参数

- layer

`keras.layers.RNN` 实例，如 `keras.layers.LSTM` 或 `keras.layers.GRU`。也可以是满足如下条件的 `keras.layers.Layer` 实例：

1. 是一个序列处理层，支持 3D+ 输入；
2. 包含 `go_backwards`, `return_sequences` 和 `return_state` 属性，与 RNN 类具有相同的语义；
3. 包含 `input_spec` 属性；

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bidirectional
