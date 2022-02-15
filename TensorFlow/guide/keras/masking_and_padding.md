# Masking and padding with Keras

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

**屏蔽**（Masking）是告诉序列处理层输入中缺少某些时间步，因此在处理数据时应该跳过这些时间步。

**填充**（padding）是一种特殊的屏蔽，其屏蔽的时间步位于序列的开头或结尾。填充是为了支持批处理，要使同一批次的所有序列长度相同，就需要填充或截断部分序列。

## 填充序列数据

在处理序列数据时，样本序列的长度不同很常见。如下（文本按单词标记化）：

```python
[
  ["Hello", "world", "!"],
  ["How", "are", "you", "doing", "today"],
  ["The", "weather", "will", "be", "nice", "tomorrow"],
]
```



## 参考

- https://www.tensorflow.org/guide/keras/masking_and_padding