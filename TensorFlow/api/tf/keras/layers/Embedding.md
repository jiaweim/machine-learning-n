# Embedding

- [Embedding](#embedding)
  - [简介](#简介)
  - [参数说明](#参数说明)
  - [输入](#输入)
  - [输出](#输出)
  - [变量位置](#变量位置)
  - [参考](#参考)

2021-12-03, 13:32
***

## 简介

嵌入层 `Embedding` 类如下：

```python
tf.keras.layers.Embedding(
    input_dim,
    output_dim,
    embeddings_initializer="uniform",
    embeddings_regularizer=None,
    activity_regularizer=None,
    embeddings_constraint=None,
    mask_zero=False,
    input_length=None,
    **kwargs
)
```

将正整数（索引）转换为固定大小的密集向量（dense vector）。例如：`[[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]`。

该层必须为模型的第一层。例如：

```python
>>> model = tf.keras.Sequential()
>>> model.add(tf.keras.layers.Embedding(1000, 64, input_length=10))
>>> # 该模型以(batch, input_length)大小的整数矩阵为输入，最大整数不超过 999 （词汇大小）
>>> # 此时 model.output_shape 为 (None, 10, 64), 其中 `None` 是 batch 维度
>>> input_array = np.random.randint(1000, size=(32, 10))
>>> model.compile('rmsprop', 'mse')
>>> output_array = model.predict(input_array)
>>> print(output_array.shape)
(32, 10, 64)
```

## 参数说明

- `input_dim`

整数，词汇大小，即最大整数索引+1.

- `output_dim`

整数，密集嵌入层维度。

- `embeddings_initializer`

`embeddings` 矩阵的初始化程序。

- `embeddings_regularizer`

应用于 `embeddings` 矩阵的正则化函数。

- `embeddings_constraint`

应用于 `embeddings` 矩阵的约束函数。

- `mask_zero`

boolean 值，是否将输入中的 0 作为特殊的填充值屏蔽掉。在使用接受可变长度输入的循环层时十分有用。

如果为 `True`，则模型中后面的 layers 需要支持屏蔽，否则抛出异常；并且索引 0 不能用于在词汇表中，此时 input_dim 大小为 vocabulary+1.

- `input_length`

输入序列长度。如果嵌入层要和 `Flatten` 及 `Dense` 层连接，就需要该参数，否则无法计算 dense 层的输出 shape。

## 输入

2D tensor with shape `(batch_size, input_length)`。

## 输出

3D tensor with shape `(batch_size, input_length, output_dim)`。

## 变量位置

如果 GPU 可用，默认将 embedding 矩阵放在 GPU，以实现最佳性能。也带来了以下问题：

- 如果使用不支持稀疏 GPU 内核的优化器（optimizer），在训练模型时会报错；
- 嵌入矩阵（embedding matrix）可能太大，GPU 放不下，此时会抛出 Out Of Memory 错误。

对这种情况，只能将嵌入矩阵放在 CPU 内存。可以定义 device scope：

```python
with tf.device('cpu:0'):
  embedding_layer = Embedding(...)
  embedding_layer.build()
```

## 参考

- https://keras.io/api/layers/core_layers/embedding/
- https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
