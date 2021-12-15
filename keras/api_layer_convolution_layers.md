# Convolution layers

- [Convolution layers](#convolution-layers)
  - [Conv1D](#conv1d)
  - [Conv2D](#conv2d)
    - [参数](#参数)
    - [示例](#示例)

2021-12-13, 17:32
***

## Conv1D

```python
tf.keras.layers.Conv1D(
    filters,
    kernel_size,
    strides=1,
    padding="valid",
    data_format="channels_last",
    dilation_rate=1,
    groups=1,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
```

一维卷积层，如时间卷积。

## Conv2D

```python
tf.keras.layers.Conv2D(
    filters,
    kernel_size,
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    groups=1,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
```

二维卷积层，如应用于图像的空间卷积。

该层创建一个卷积核，卷积核与输入层进行卷积运算，产生一个输出张量：

- 如果 `use_bias=True`，则创建一个 bias 向量，添加到输出中；
- 如果 `activation` 不为 `None`，则应用到输出。

如果将 `Conv2D` 作为第一层，则应该提供关键字参数 `input_shape`，例如，对数据格式为 `data_format="channels_last"` 的 128x128 RGB 图像，其 `input_shape=(128, 128, 3)`。如果图像 dimension 可变，则设置 `input_shape=None`。

### 参数




### 示例

```python
>>> # The inputs are 28x28 RGB images with `channels_last` and the batch
>>> # size is 4.
>>> input_shape = (4, 28, 28, 3)
>>> x = tf.random.normal(input_shape)
>>> y = tf.keras.layers.Conv2D(
... 2, 3, activation='relu', input_shape=input_shape[1:])(x)
>>> print(y.shape)
(4, 26, 26, 2)
```
