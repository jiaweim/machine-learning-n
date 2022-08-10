# Layer

## 简介

```python
tf.keras.layers.Layer(
    trainable=True, name=None, dtype=None, dynamic=False, **kwargs
)
```

所有 layer 的基类。

## 方法

### get_weights

Last updated: 2022-08-09, 14:09

```python
get_weights()
```

以 NumPy 数组的形式返回 layer 的当前权重。

layer 的权重表示其状态。该函数以 list of NumPy 数组的形式同时返回与该 layer 相关的 trainable 和 non-trainable 权重值。

例如，`Dense` layer 返回包含两个值的列表：kernel matrix 和 bias vector。这些权重可用于初始化其它 `Dense` layer:

```python
>>> layer_a = tf.keras.layers.Dense(1, kernel_initializer=tf.constant_initializer(1.0))
>>> a_out = layer_a(tf.convert_to_tensor([[1.0, 2.0, 3.0]]))
>>> layer_a.get_weights()
[array([[1.],
        [1.],
        [1.]], dtype=float32),
 array([0.], dtype=float32)]

>>> layer_b = tf.keras.layers.Dense(1, kernel_initializer=tf.constant_initializer(2.0))
>>> b_out = layer_b(tf.convert_to_tensor([[10.0, 20.0, 30.0]]))
>>> layer_b.get_weights()
[array([[2.],
        [2.],
        [2.]], dtype=float32),
 array([0.], dtype=float32)]

>>> layer_b.set_weights(layer_a.get_weights())
>>> layer_b.get_weights()
[array([[1.],
        [1.],
        [1.]], dtype=float32),
 array([0.], dtype=float32)]
```

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer
