# RNN

- [RNN](#rnn)
  - [简介](#简介)
  - [设置](#设置)
  - [内置 RNN 层](#内置-rnn-层)
  - [输出和状态](#输出和状态)
  - [LSTM 和 GRU](#lstm-和-gru)
  - [参考](#参考)

2021-12-03, 17:19
***

## 简介

循环神经网络擅长对序列数据，如时间序列、自然语言进行建模。

从原理上讲，RNN 使用一个 for 循环来迭代一个序列的时间段（timestep），同时维护一个内部状态，编码到目前为止看到的所有的时间段信息。

Keras RNN API 设计的要点：

- 易用性：内置的 `keras.layers.RNN`, `keras.layers.LSTM`, `keras.layers.GRU` 可用于快速构建循环模型；
- 自定义方便：可以定义自己的 RNN cell layer（for 循环内部），并将其与通用的 `keras.layers.RNN` layer 一起会用。

## 设置

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

## 内置 RNN 层

在 Keras 中有三个内置的 RNN 层：

1. `keras.layers.SimpleRNN`，全连接 RNN，前一个 timestep 的输出作为下一个 timestep 的输入。
2. `keras.layers.GRU`，在 [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) 首次提出；
3. `keras.layers.LSTM`，在 [Hochreiter & Schmidhuber, 1997](https://www.bioinf.jku.at/publications/older/2604.pdf) 首次提出。

2015 年初，Keras 有了第一个可复用的 LSTM 和 GRU 开源 Python 实现。

下面是一个处理整数序列的简单 `Sequential` 模型子，将每个整数嵌入一个 64 维的向量，然后用 LSTM 层处理向量序列。

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential()
# 添加嵌入层，期望输入词汇量为 1000，输出嵌入维度为 64
model.add(layers.Embedding(input_dim=1000, output_dim=64))
# 添加 LSTM 层，包含 128 个内部单元
model.add(layers.LSTM(128))
# 添加包含 10 个 单元的 Dense 层
model.add(layers.Dense(10))
model.summary()
```

```txt
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

内置 RNN 支持许多功能：

- 通过 `dropout` 和 `recurrent_dropout` 参数支持 Recurrent dropout；
- 通过 `go_backwards` 参数支持反向输入序列；
- 通过 `unroll` 参数支持循环展开（在 CPU 上处理短序列时，能提升速度）。

## 输出和状态

RNN 层的输出默认包含每个样本的单个向量。该向量对应于最后一个 timestep 的 RNN cell 输出，包含整个输入序列的信息。输出 shape 为 `(batch_size, units)`，其中 `units` 和 `units` 参数对应。

如果设置 `return_sequences=True`，RNN 层可以返回每个样本的整个序列输出（one vector per timestep per sample）。此时输出 shape 为 `(batch_size, timesteps, units)`。

```python
model = keras.Sequential()
model.add(layers.Embedding(input_dim=1000, output_dim=64))

# The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
model.add(layers.GRU(256, return_sequences=True))

# The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
model.add(layers.SimpleRNN(128))

model.add(layers.Dense(10))

model.summary()
```

```txt
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

## LSTM 和 GRU

`SimpleRNN` 过于简化，没有实用价值，其最大问题在于，在时刻 t，理论上它应该能够记住许多时间步之前见过的信息，但实际上它不可能学到这种长期依赖，其原因在于梯度消失问题（vanishing gradient problem）：随着层数的增加，网络最终变得无法训练。LSTM 层和 GRU 层为了解决这个问题而设计。

## 参考

- https://keras.io/guides/working_with_rnns/
