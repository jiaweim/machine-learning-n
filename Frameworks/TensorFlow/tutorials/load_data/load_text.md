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
data_url = 'https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz'
dataset_dir = utils.get_file(
    origin=data_url,
    untar=True,
    cache_dir = 'stack_overflow',
    cache_subdir=''
)
dataset_dir = pathlib.Path(dataset_dir).parent
```

查看目录：

```python
list(dataset_dir.iterdir())
```

```sh
[WindowsPath('stack_overflow/README.md'),
 WindowsPath('stack_overflow/stack_overflow_16k.tar.gz'),
 WindowsPath('stack_overflow/test'),
 WindowsPath('stack_overflow/train')]
```

```python
train_dir = dataset_dir / "train"
list(train_dir.iterdir())
```

```sh
[WindowsPath('stack_overflow/train/csharp'),
 WindowsPath('stack_overflow/train/java'),
 WindowsPath('stack_overflow/train/javascript'),
 WindowsPath('stack_overflow/train/python')]
```

`train/csharp`, `train/java`, `train/python` 和 `train/javascript` 目录包含许多文本文件，每个文本文件都是一个 Stack Overflow 问题。

打印一个样本看看：

```python
sample_file = train_dir / "python/1755.txt"
with open(sample_file) as f:
    print(f.read())
```

```sh
why does this blank program print true x=true.def stupid():.    x=false.stupid().print x
```

### 加载数据集

下面从磁盘加载数据集，并将其处理成适合训练的格式。为此，我们使用 `tf.keras.utils.text_dataset_from_directory` 创建带标签数据集 `tf.data.Dataset`。

## 参考

- https://www.tensorflow.org/tutorials/load_data/text
