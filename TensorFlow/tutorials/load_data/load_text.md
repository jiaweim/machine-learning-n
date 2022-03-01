# 载入文本

2022-03-01, 16:11
***

## 简介

下面介绍加载和预处理文本的两种方法：

- 使用 keras 工具和预处理层。包括将数据转换为 `tf.data.Dataset` 的 `tf.keras.utils.text_dataset_from_directory` 以及用于数据标准化、标记化和矢量化的 `tf.keras.layers.TextVectorization`。对新手，建议使用该方法。
- 使用底层 API，如 `tf.data.TextLineDataset` 加载文本文件，用 TensorFlow Text APIs 如 `text.UnicodeScriptTokenizer` 和 `text.case_fold_utf8` 等更细粒度地预处理数据。

```python
import collections
import pathlib

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import utils
from tensorflow.keras.layers import TextVectorization

import tensorflow_datasets as tfds
import tensorflow_text as tf_text
```

## 实例 1：预测 Stack Overflow 问题的标签

首先，从 Stack Overflow 下载编程问题的数据集。每个问题只有一个标记，任务时开发一个模型来预测问题的标签。这是一个多分类问题。

### 下载并查看数据集

首先使用 `tf.keras.utils.get_file` 下载 Stack Overflow 数据集，并查看数据的目录结构：

```python

```


## 参考

- https://www.tensorflow.org/tutorials/load_data/text
