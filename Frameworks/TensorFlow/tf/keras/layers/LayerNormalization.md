# tf.keras.layers.LayerNormalization

2022-03-07, 21:01
***

## 简介

```python
tf.keras.layers.LayerNormalization(
    axis=-1, epsilon=0.001, center=True, scale=True,
    beta_initializer='zeros', gamma_initializer='ones',
    beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
    gamma_constraint=None, **kwargs
)
```

Layer normalization layer.

Layer norm 将每个样本的数据，归一化为均值为 0，方差为 1.
而 batch norm 将批量样本的每个特征的数据，归一化为均值为 0，方差为 1.

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization
