# layers.Layer

- [layers.Layer](#layerslayer)
  - [Layer](#layer)
  - [Dense](#dense)

2021-06-04, 09:34
***

## Layer

所有（网络）层的父类。

```python
tf.keras.layers.Layer(
    trainable=True, name=None, dtype=None, dynamic=False, **kwargs
)
```

Layer 是一个可调用对象，接受一个或多个 tensors 输入，并输出一个或多个 tensors。 `Layer` 涉及的计算放在 `call()` 方法中，状态（权重变量）定义在构造函数 `__init__()` 或 `build()` 方法汇总。

## Dense

定义全连接 NN 层。

```python
tf.keras.layers.Dense(
    units, activation=None, use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    bias_constraint=None, **kwargs
)
```

`Dense` 实现了如下操作： `output = activation(dot(input, kernel) + bias)` ，其中 `activation` 是 element-wise 激活函数， `kernel` 是权重 matrix， `bias` 是 bias 向量。

此外，layer 的属性在调用后无法修改（ `trainable` 属性除外）。

实例：

```python
# Create a `Sequential` model and add a Dense layer as the first layer.
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(16,)))
model.add(tf.keras.layers.Dense(32, activation='relu'))
# Now the model will take as input arrays of shape (None, 16)
# and output arrays of shape (None, 32).
# Note that after the first layer, you don't need to specify
# the size of the input anymore:
model.add(tf.keras.layers.Dense(32))
model.output_shape
```

| 参数 | 说明 |
| --- | --- |
| units | 正整数，输出空间维数 |
| activation | 激活函数 |
| use-bias | Boolean，是否使用 bias 向量 |
| kernel_initializer | 初始化 `kernel` 权重矩阵的方法 |
| bias_initializer | 初始化 `bias` 向量的方法 |
| kernel_regularizer | 应用于 `kernel` 权重矩阵的正则函数 |
| bias_regularizer | 应用于 `bias` 向量的正则函数 |
| activity_regularizer | 应用于该layer输出值的正则函数 |
| kernel_constraint | 应用于 `kernel` 权重矩阵的约束函数 |
| bias_constraint | 应用于 `bias` 向量的约束函数 |

**输入 shape:**
shape 为 `(batch_size, ..., input_dim)` N-D tensor ，一般是为 2D shape `(batch_size, input_dim)` .

**输出 shape:**
shape 为 `(batch_size, ..., units)` 的 N-D tensor。例如，对 2D 输入 `(batch_size, input_dim)` ，输出一般为 `(batch_size, units)` 。
