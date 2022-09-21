# Layer

***

## 简介

```python
tf.keras.layers.Layer(
    trainable=True, name=None, dtype=None, dynamic=False, **kwargs
)
```

所有 layer 的基类。

## 方法

### add_loss

```python
add_loss(
    losses, **kwargs
)
```

添加 loss 张量。

有些 loss（如正则化 loss）可能依赖于调用 layer 时传入的输入。因此，当在不同输入如 `a` 和 `b` 上使用相同的 layer，`layer.losses` 中可能有些依赖于 `a`，有些依赖于 `b`。此方法会自动跟踪依赖项。

- `add_loss` 可以在 subclass layer 或模型的 `call` 函数中使用，此时 `losses` 为 Tensor 或 Tensor list。

**例如：**

```python
class MyLayer(tf.keras.layers.Layer):
  def call(self, inputs):
    self.add_loss(tf.abs(tf.reduce_mean(inputs)))
    return inputs
```

- `add_loss` 也可以在使用函数 API 构建模型时直接调用，此时，传递给模型的任何 loss 张量必须是符号性的，并且能够追溯到模型的输入。这些 loss 称为模块拓扑的一部分，并在 `get_config` 中追踪。

**例如：**

```python
inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(10)(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)
# Activity regularization.
model.add_loss(tf.abs(tf.reduce_mean(x)))
```

如果所需 loss 不是这种情况，例如需要引用模型中一个 layer 的变量计算，则可以将 loss 包装在一个零参数 lambda 中。这种 loss 不会作为模型拓扑的一部分进行追踪，无法序列化。

### build

Last updated: 2022-09-21, 11:16

```python
build(
    input_shape
)
```

创建 layer 的变量，用于 subclass 实现中，以推迟 weight 的创建。

使用继承实现 **Layer** 或 **Model** 时，覆盖该方法可以将状态（weight）的创建从 layer 的实例化解耦出来，`build()`方法会在第一次调用 `call()` 时被自动调用。

该方法一般用在通过继承创建 Layer。

|参数|说明|
|---|---|
|input_shape|`TensorShape` 实例，或 `TensorShape` 实例列表|

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
