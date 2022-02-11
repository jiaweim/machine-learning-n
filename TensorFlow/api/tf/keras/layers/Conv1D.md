# Conv1D

2022-02-10, 19:53
***

## 简介

1D 卷积层。

```python
tf.keras.layers.Conv1D(
    filters, kernel_size, strides=1, padding='valid',
    data_format='channels_last', dilation_rate=1, groups=1,
    activation=None, use_bias=True, kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    bias_constraint=None, **kwargs
)
```

## 参数

### padding

可选值：

- "valid", no padding
- "same", 边界均匀添加 padding，使得输出和输入的 height/width 相同
- "causal", 



## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D
