# Dense

- [Dense](#dense)
  - [简介](#简介)
  - [参数](#参数)
  - [示例](#示例)
  - [参考](#参考)

2022-02-11, 13:55
****

## 简介

```python
tf.keras.layers.Dense(
    units,
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
```

常规的全连接 NN 层。

`Dense` 实现了 $output=activation(dot(input, kernel)+bias)$ 操作，其中

- `activation` 是 element-wise 激活函数，以 `activation` 参数设置；
- `kernel` 是由 layer 创建的权重矩阵；
- `bias` 是由 layer 创建的 bias 向量（必须设置 `use_bias` 为 `True`）。

它们都是 `Dense` 的属性。

> **NOTE**
> 如果输入 $rank>2$，则 `Dense` 沿着 **inputs** 的最后一个轴和 `kernel` 的第 0 轴计算点乘（`tf.tensordot`）。例如，如果输入为 `(batch_size, d0, d1)`，则创建的 `kernel` 为 `(d1, units)`，`kernel` 沿着输入的 axis 2 操作，即对每个 shape 为 `(1, 1, d1)` 的子张量（共 `batch_size * d0` 个）进行操作。输出为 `(batch_size, d0, units)`
> `TimeDistributed(Dense(...))` 和 `Dense(...)` 等价。

另外，layer 被调用之后，其属性不能更改（`trainable` 属性除外）。如果传入 `input_shape` kwarg 参数，keras 会在当前层前面创建一个输入层。和显式定义 `InputLayer` 等价。

## 参数

|参数|说明|
|---|---|
|units|正整数，输出空间维度|

- **activation**

激活函数。不指定表示不使用激活函数（即线性激活：$a(x)=x$）

|use_bias|boolean，是否使用 bias 向量|
|kernel_initializer|`kernel` 权重矩阵的初始化器|
|bias_initializer|bias 向量初始化器|
|kernel_regularizer|`kernel` 权重矩阵正则化函数|
|bias_regularizer|bias 向量正则化函数|
|activity_regularizer|应用于该层输出值（激活值）的正则化函数|
|kernel_constraint|应用于 `kernel` 权重矩阵的约束函数|
|bias_constraint|应用于 bias 向量的约束函数|

**输入 shape**

`(batch_size, ..., input_dim)` 的 N 维张量。最常见的是 shape 为 `(batch_size, input_dim)` 的 2D 输入向量。

**输出 shape**

输出 shape 为 `(batch_size, ..., units)` 的 N 维张量。例如，如果输入为 `(batch_size, input_dim)`，则输出为 `(batch_size, units)`。

## 示例

```python
>>> # 创建 `Sequential` 模型，`Dense` 作为第一层
>>> model = tf.keras.models.Sequential()
>>> model.add(tf.keras.Input(shape=(16,)))
>>> model.add(tf.keras.layers.Dense(32, activation='relu'))
>>> # 此时模型输入 shape (None, 16)
>>> # 输出 shape (None, 32).
>>> # 第一层后，不需要指定输入 shape
>>> model.add(tf.keras.layers.Dense(32))
>>> model.output_shape
(None, 32)
```

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
