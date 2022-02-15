# Dense

2022-02-11, 13:55
****

## 简介

常规的全连接 NN 层。

```python
tf.keras.layers.Dense(
    units, activation=None, use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    bias_constraint=None, **kwargs
)
```

`Dense` 实现了 $output=activation(dot(input, kernel)+bias)$ 操作，其中

- `activation` 是 element-wise 激活函数，以 `activation` 参数设置；
- `kernel` 是由 layer 创建的权重矩阵；
- `bias` 是由 layer 创建的 bias 向量（必须设置 `use_bias` 为 `True`）。

> 如果输入的 $rank>2$，则 `Dense` 沿着输入的最后一个轴和 `kernel` 的第 0 轴计算点乘（`tf.tensordot`）。例如，如果输入为 `(batch_size, d0, d1)`，则创建的 `kernel` 为 `(d1, units)`，`kernel` 沿着输入的第 axis 2 操作，即对每个 shape 为 `(1, 1, d1)` 的子张量（共 `batch_size * d0` 个）进行操作。输出为 `(batch_size, d0, units)`

另外，layer 被调用之后，其属性不能更改（`trainable` 属性除外）。如果传入 `input_shape` kwarg 参数，keras 会在当前层前面创建一个输入层。和显式定义 `InputLayer` 等价。

例如：

```python
>>> # Create a `Sequential` model and add a Dense layer as the first layer.
>>> model = tf.keras.models.Sequential()
>>> model.add(tf.keras.Input(shape=(16,)))
>>> model.add(tf.keras.layers.Dense(32, activation='relu'))
>>> # Now the model will take as input arrays of shape (None, 16)
>>> # and output arrays of shape (None, 32).
>>> # Note that after the first layer, you don't need to specify
>>> # the size of the input anymore:
>>> model.add(tf.keras.layers.Dense(32))
>>> model.output_shape
(None, 32)
```

## 参数

|参数|说明|
|---|---|
|units|Positive integer, dimensionality of the output space|
|activation|Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: $a(x) = x)$|
|use_bias|Boolean, whether the layer uses a bias vector|
|kernel_initializer|Initializer for the kernel weights matrix|
|bias_initializer|Initializer for the bias vector|
|kernel_regularizer|Regularizer function applied to the kernel weights matrix|
|bias_regularizer|Regularizer function applied to the bias vector.|
|activity_regularizer|Regularizer function applied to the output of the layer (its "activation")|
|kernel_constraint|Constraint function applied to the kernel weights matrix.|
|bias_constraint|Constraint function applied to the bias vector|

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
