# RNN with Keras

- [RNN with Keras](#rnn-with-keras)
  - [简介](#简介)
  - [配置](#配置)
  - [内置 RNN 层](#内置-rnn-层)
  - [输出和状态](#输出和状态)
  - [RNN 层和 RNN 单元](#rnn-层和-rnn-单元)
  - [跨批量状态](#跨批量状态)
    - [RNN State Reuse](#rnn-state-reuse)
  - [双向 RNN](#双向-rnn)
  - [性能优化和 CuDNN 核](#性能优化和-cudnn-核)
    - [使用 CuDNN 内核](#使用-cudnn-内核)
  - [参考](#参考)

2022-03-11, 22:35
***

## 简介

循环神经网络（Recurrent neural network, RNN）是一类擅长对时间序列或自然语言等序列数据建模的神经网络。

实践中，RNN 层使用 `for` 循环迭代序列的时间步，同时维护一个内部状态，该状态对看过的时间步信息进行编码。

> 内部状态不是 weight，而是记录序列、时间步等状态信息。

Keras RNN API 的特点：

- 使用简单：内置的 `keras.layers.RNN`, `keras.layers.LSTM` 和 `keras.layers.GRU` 不需要复杂配置，就可用于构建 RNN 模型；
- 自定义简单：可以自定义 RNN cell 层（`for` 循环内部），并将其与通用 `keras.layers.RNN` 层(`for` 循环本身)一起使用，就可以构建自定义 RNN 模型。

## 配置

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

## 内置 RNN 层

Keras 有三个内置的 RNN 层：

1. `keras.layers.SimpleRNN`，全连接 RNN，上一个时间步的输出送到下一个时间步；
2. `keras.layers.GRU`
3. `keras.layers.LSTM`

在 2015 年初，Keras 推出了第一个可重用的 LSTM 和 GRU 的开源 Python 实现。

下面是一个简单的 `Sequential` 模型，该模型处理整数序列，将每个整数嵌入到 64 维向量中，然后在 `LSTM` 层中处理：

```python
model = keras.Sequential()
# 添加嵌入层，要求输入词汇量为 1000，输出嵌入维度 64
model.add(layers.Embedding(input_dim=1000, output_dim=64))

# 添加包含 128 个内部单元的 LSTM 层
model.add(layers.LSTM(128))

# 添加包含 10 个单元的 Dense 层
model.add(layers.Dense(10))

model.summary()
```

```sh
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, None, 64)          64000     
_________________________________________________________________
lstm (LSTM)                  (None, 128)               98816     
_________________________________________________________________
dense (Dense)                (None, 10)                1290      
=================================================================
Total params: 164,106
Trainable params: 164,106
Non-trainable params: 0
_________________________________________________________________
```

内置 RNN 支持许多特性：

- 通过 `dropout` 和 `recurrent_dropout` 参数提供循环 dropout 功能；
- 通过 `go_backwards` 参数提供反向处理输入序列的功能；
- 通过 `unroll` 参数展开循环，在 CPU 上处理短序列时可以大大提高速度。

## 输出和状态

RNN 层对每个样本默认输出一个向量，对应最后一个时间步 RNN cell 的输出。输出 shape 为 `(batch_size, units)`，其中 `units` 和 RNN 层构造函数的 `units` 参数一致。

如果设置 `return_sequences=True`，RNN 层可以返回整个输出序列，即每个时间步对应一个向量。此时输出 shape 为 `(batch_size, timesteps, units)`。

```python
model = keras.Sequential()
model.add(layers.Embedding(input_dim=1000, output_dim=64))

# GRU 的输出为 3D 张量，shape 为 (batch_size, timesteps, 256)
model.add(layers.GRU(256, return_sequences=True))

# SimpleRNN 输出 2D 张量，shape 为 (batch_size, 128)
model.add(layers.SimpleRNN(128))

model.add(layers.Dense(10))

model.summary()
```

```sh
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, None, 64)          64000     
_________________________________________________________________
gru (GRU)                    (None, None, 256)         247296    
_________________________________________________________________
simple_rnn (SimpleRNN)       (None, 128)               49280     
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290      
=================================================================
Total params: 361,866
Trainable params: 361,866
Non-trainable params: 0
_________________________________________________________________
```

另外，RNN 层可以返回其内部的最终状态。返回的内部状态可用于恢复 RNN 状态，也可以用来初始化另一个 RNN。该设置通常用在 encoder-decoder seq2seq 模型，其中 encoder 的最终状态用于 decoder 的初始状态。

将 `return_state` 设置为 `True` 使 RNN 返回其内部状态。注意 LSTM 有 2 个状态张量，而 `GRU` 只有一个。

使用 `initial_state` 配置 RNN 层的初始状态。注意内部状态的 shape 要和 layer 的 unit size 匹配，如下:

```python
encoder_vocab = 1000
decoder_vocab = 2000

encoder_input = layers.Input(shape=(None,))
encoder_embedded = layers.Embedding(input_dim=encoder_vocab, output_dim=64)(
    encoder_input
)

# 除了输出，还返回状态
output, state_h, state_c = layers.LSTM(64, return_state=True, name="encoder")(
    encoder_embedded
)
encoder_state = [state_h, state_c]

decoder_input = layers.Input(shape=(None,))
decoder_embedded = layers.Embedding(input_dim=decoder_vocab, output_dim=64)(
    decoder_input
)

# 将 2 个状态作为新 LSTM 层的初始状态
decoder_output = layers.LSTM(64, name="decoder")(
    decoder_embedded, initial_state=encoder_state
)
output = layers.Dense(10)(decoder_output)

model = keras.Model([encoder_input, decoder_input], output)
model.summary()
```

```sh
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, None)]       0                                            
__________________________________________________________________________________________________
input_2 (InputLayer)            [(None, None)]       0                                            
__________________________________________________________________________________________________
embedding_2 (Embedding)         (None, None, 64)     64000       input_1[0][0]                    
__________________________________________________________________________________________________
embedding_3 (Embedding)         (None, None, 64)     128000      input_2[0][0]                    
__________________________________________________________________________________________________
encoder (LSTM)                  [(None, 64), (None,  33024       embedding_2[0][0]                
__________________________________________________________________________________________________
decoder (LSTM)                  (None, 64)           33024       embedding_3[0][0]                
                                                                 encoder[0][1]                    
                                                                 encoder[0][2]                    
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 10)           650         decoder[0][0]                    
==================================================================================================
Total params: 258,698
Trainable params: 258,698
Non-trainable params: 0
__________________________________________________________________________________________________
```

## RNN 层和 RNN 单元

除了内置 RNN 层，RNN API 还提供 cell-level API。与处理批量输入序列的 RNN 层不同，RNN cell 只处理单个时间步。

cell 是 RNN 层 `for` 循环内部部分。将 cell 包裹在 `keras.layers.RNN` 中，就获得能够处理批量序列的 layer，例如 `RNN(LSTMCell(10))`。

在数学上，`RNN(LSTMCell(10))` 与 `LSTM(10)` 生成的结果相同。实际上，TF v1.x 中的 `LSTM` 层就是用 `RNN` 层包裹 `LSTMCell` 实现的。不过，内置的 `GRU` 和 `LSTM` 层可以使用 CuDNN，从而提高性能。

有三个内置的 RNN cell，与三个 RNN 层对应：

- `keras.layers.SimpleRNNCell`
- `keras.layers.GRUCell`
- `keras.layers.LSTMCell`

cell 结合 `keras.layers.RNN` 类，可以很容易自定义 RNN 框架。

## 跨批量状态

在处理非常长的序列时，可能需要使用跨批量状态（cross-batch statefullness）模式。

通常情况下，RNN 层的内部状态在每次看到一个新的批次时重置（即认为该层看到的每个样本独立于过去）。在处理给定的样本时，RNN 层只维护一个状态。

如果你有很长的序列，则将该序列分解成短序列，把这些短序列按顺序输入 RNN 层，且不重置层的状态，这样 RNN 层可以保留整个序列的信息，即时它一次只能看到一个子序列。

只需要在构造函数中设置 `stateful=True` 就可以做到这一点。

假设你有一个长序列 `s = [t0, t1, ... t1546, t1547]`，可以将其分成：

```python
s1 = [t0, t1, ... t100]
s2 = [t101, ... t201]
...
s16 = [t1501, ... t1547]
```

然后你可以通过以下方式处理它：

```python
lstm_layer = layers.LSTM(64, stateful=True)
for s in sub_sequences:
  output = lstm_layer(s)
```

可以使用 `layer.reset_states()` 清除状态。

下面是完整示例：

```python
paragraph1 = np.random.random((20, 10, 50)).astype(np.float32)
paragraph2 = np.random.random((20, 10, 50)).astype(np.float32)
paragraph3 = np.random.random((20, 10, 50)).astype(np.float32)

lstm_layer = layers.LSTM(64, stateful=True)
output = lstm_layer(paragraph1)
output = lstm_layer(paragraph2)
output = lstm_layer(paragraph3)

# reset_states() 将缓存的状态重置为初始状态。
# 如果没有提供initial_state，则默认使用 zero-states.
lstm_layer.reset_states()
```

### RNN State Reuse

`layer.weights()` 不包含RNN 层的状态。如果想重用 RNN 层的状态，可以使用 `layer.states` 获得状态，然后通过 keras 函数 API，如 `new_layer(inputs, initial_state=layer.states)` 设置新层的状态，也可以使用继承 Model API。

需要注意的是，这种情况不能使用 sequential 模型，因为 sequential 模型只支持单个输入，不支持额外的 `initial_state` 输入。

```python
paragraph1 = np.random.random((20, 10, 50)).astype(np.float32)
paragraph2 = np.random.random((20, 10, 50)).astype(np.float32)
paragraph3 = np.random.random((20, 10, 50)).astype(np.float32)

lstm_layer = layers.LSTM(64, stateful=True)
output = lstm_layer(paragraph1)
output = lstm_layer(paragraph2)

existing_state = lstm_layer.states

new_lstm_layer = layers.LSTM(64)
new_output = new_lstm_layer(paragraph3, initial_state=existing_state)
```

## 双向 RNN

对时间序列以外的序列，例如文本，如果 RNN 模型不仅从头到尾处理序列，还从后向前处理序列，性能往往更好。例如，要预测句子的下一个单词，如果不仅知道前面的单词，还之后后面的单词，往往效果更好。

Keras 提供了一个简单的 API 来构建这样的双向 RNN：`keras.layers.Bidirectional` wrapper。

```python
model = keras.Sequential()

model.add(
    layers.Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=(5, 10))
)
model.add(layers.Bidirectional(layers.LSTM(32)))
model.add(layers.Dense(10))

model.summary()
```

```sh
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
bidirectional (Bidirectional (None, 5, 128)            38400     
_________________________________________________________________
bidirectional_1 (Bidirection (None, 64)                41216     
_________________________________________________________________
dense_3 (Dense)              (None, 10)                650       
=================================================================
Total params: 80,266
Trainable params: 80,266
Non-trainable params: 0
_________________________________________________________________
```

`Bidirectional` 会复制传入的 RNN 层，并翻转新复制层的 `go_backwards` 字段，使其以相反的顺序处理输入。

`Bidirectional` RNN 默认将正向和反向层输出串联。如果需要不同的合并方式，可以设置 `Bidirectional` 的 `merge_mode` 参数。

## 性能优化和 CuDNN 核

在 TensorFlow 2.0 中，内置的 LSTM 和 GRU 在 GPU 可用时，默认利用 cuDNN 内核。之前的 `keras.layers.CuDNNLSTM/CuDNNGRU` 已弃用。

由于 CuDNN 内核实在一定的前提条件下构建的，如果改变了内置的 LSTM 或 GRU 层的默认参数，可能无法使用 CuDNN 内核。例如：

- 将 `activation` 从 `tanh` 换成其它选项；
- 将 `recurrent_activation` 从 `sigmoid` 换成其它选项；
- 使用 `recurrent_dropout` > 0；
- 将 `unroll` 设置为 True，迫使 LSTM/GRU 将内部 `tf.while_loop` 转换为展开的 `for` 循环；
- `use_bias` 设置为 False；
- 当输入数据不是严格右填充时使用 masking（如果 mask 对应严格右填充，CuDNN 仍然可用）

### 使用 CuDNN 内核

下面构建一个简单的 LSTM 模型来演示性能差异。

我们使用 MNIST 数字的行作为输入序列（将图片的每一行像素作为一个时间步），然后预测数字的标签：

```python
batch_size = 64
# 每个 MNIST 图像 batch 张量 shape 为 (batch_size, 28, 28).
# 每个输入序列 size 为 (28, 28) (图像高度作为时间步处理).
input_dim = 28

units = 64
output_size = 10  # 输出标签 0 to 9

# 构建 RNN 模型
def build_model(allow_cudnn_kernel=True):
    # CuDNN 只能在 layer 层次使用，cell 层次不能
    # 即 `LSTM(units)` 可以使用 CuDNN 内核
    # 而 `RNN(LSTMCell(units))` 只能在 non-CuDNN 内核运行
    if allow_cudnn_kernel:
        # 使用默认选项的 LSTM 能用 CuDNN
        lstm_layer = keras.layers.LSTM(units, input_shape=(None, input_dim))
    else:
        # 使用 RNN 层包装的 LSTMCell 不能用 CuDNN
        lstm_layer = keras.layers.RNN(
            keras.layers.LSTMCell(units), input_shape=(None, input_dim)
        )
    model = keras.models.Sequential(
        [
            lstm_layer,
            keras.layers.BatchNormalization(),
            keras.layers.Dense(output_size),
        ]
    )
    return model
```

载入 MNIST 数据集：

```python
mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
sample, sample_label = x_train[0], y_train[0]
```

开始创建并训练模型。

由于是分类模型，我们选择 `sparse_categorical_crossentropy` 作为损失函数。模型输出 shape 为 `[batch_size, 10]`。模型的目标是整数向量，整数范围 0-9.

```python
model = build_model(allow_cudnn_kernel=True)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="sgd",
    metrics=["accuracy"],
)


model.fit(
    x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=1
)
```

```sh
938/938 [==============================] - 13s 11ms/step - loss: 0.9520 - accuracy: 0.6985 - val_loss: 0.6148 - val_accuracy: 0.7913
<keras.callbacks.History at 0x7f9583a332d0>
```

然后和不使用 CuDNN 内核的模型对比：

```python
noncudnn_model = build_model(allow_cudnn_kernel=False)
noncudnn_model.set_weights(model.get_weights())
noncudnn_model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="sgd",
    metrics=["accuracy"],
)
noncudnn_model.fit(
    x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=1
)
```

```sh
938/938 [==============================] - 82s 86ms/step - loss: 0.3866 - accuracy: 0.8838 - val_loss: 0.3223 - val_accuracy: 0.8965
<keras.callbacks.History at 0x7f9584805550>
```

在 GPU 机器上运行，使用 CuDNN 构建的模型比常规 TensorFlow 内核的模型训练要快。



## 参考

- https://www.tensorflow.org/guide/keras/rnn
