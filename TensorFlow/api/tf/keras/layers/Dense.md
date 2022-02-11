# Dense

2022-02-11, 13:55
***

## 简介

```python
tf.keras.layers.Dense(
    units, activation=None, use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    bias_constraint=None, **kwargs
)
```

## 参数

**activation**

使用的激活函数。如果未指定，则不使用任何激活，即线性激活 "linear" $a(x)=x$。


## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
