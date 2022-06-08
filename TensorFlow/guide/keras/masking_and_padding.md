# 用 Keras 屏蔽和填充序列

- [用 Keras 屏蔽和填充序列](#用-keras-屏蔽和填充序列)
  - [配置](#配置)
  - [简介](#简介)
  - [填充序列数据](#填充序列数据)
  - [屏蔽](#屏蔽)
  - [参考](#参考)

2022-02-15, 13:55
***

## 配置

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

## 简介

**屏蔽**（Masking）是告诉序列处理层输入中缺少某些时间步，在处理数据时应该跳过这些时间步。

**填充**（Padding）是一种特殊的屏蔽，其屏蔽的时间步位于序列的开头或结尾。填充是为了支持批处理，为了使同一批次的所有序列长度相同，需要填充或截断部分序列。

## 填充序列数据

在处理序列数据时，样本序列的长度不同很常见。例如（文本按单词标记化）：

```python
[
  ["Hello", "world", "!"],
  ["How", "are", "you", "doing", "today"],
  ["The", "weather", "will", "be", "nice", "tomorrow"],
]
```

经过词汇表转换为整数向量：

```py
[
  [71, 1331, 4231]
  [73, 8, 3215, 55, 927],
  [83, 91, 1, 645, 1253, 927],
]
```

此时数据是一个嵌套列表，样本长度分别为 3, 5 和 6。由于深度学习模型要求输入数据必须是单个张量（在本例中 shape 为 `(batch_size, 6, vocab_size)`），对长度小于最长样本的样本，需要填充一些占位符（也可以在填充短样本前截断长样本）。

Keras 使用 [tf.keras.utils.pad_sequences](../../api/tf/keras/utils/pad_sequences.md) 截断和填充 Python 列表到一个指定长度。例如：

```py
raw_inputs = [
    [711, 632, 71],
    [73, 8, 3215, 55, 927],
    [83, 91, 1, 645, 1253, 927],
]

# 默认填充 0，可以使用 `value` 参数设置填充值
# `padding` 为 "pre" (在开头填充) 或 "post" (在末尾填充)
# 在 RNN 中推荐使用 "post" 填充，这样才能使用 CuDNN 实现的 layers
padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(
    raw_inputs, padding="post"
)
print(padded_inputs)
```

```sh
[[ 711  632   71    0    0    0]
 [  73    8 3215   55  927    0]
 [  83   91    1  645 1253  927]]
```

## 屏蔽



## 参考

- https://www.tensorflow.org/guide/keras/masking_and_padding