# 基本文本分类

- [基本文本分类](#基本文本分类)
  - [简介](#简介)
  - [情感分析](#情感分析)
    - [下载 IMDB 数据集](#下载-imdb-数据集)
    - [加载数据集](#加载数据集)
  - [参考](#参考)

2022-01-01, 22:10
***

## 简介

下面演示如何对文本进行分类，即训练一个二分类网络，对 IMDB 数据集进行情感分析（sentiment analysis）。

```python
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
```

```python
print(tf.__version__)
```

```sh
2.8.0
```

## 情感分析

下面训练一个情感分析模型，根据电影评论文本将评论分好评和差评。这是一个典型的二元分类问题。

下面使用大型电影评论数据集，该数据集包含来自[互联网电影数据库](https://www.imdb.com/) 的 50,000 个电影评论的文本。这些评估分为 25,000 个训练样本和 25,000 个测试样本。训练样本和测试样本都是平衡的，即包含相同数量的好评和差评。

### 下载 IMDB 数据集

下载并解压数据集：

```python
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file("aclImdb_v1", url,
                                    untar=True, cache_dir='.',
                                    cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
```

```sh
Downloading data from https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
84131840/84125825 [==============================] - 3s 0us/step
84140032/84125825 [==============================] - 3s 0us/step
```

```python
os.listdir(dataset_dir)
```

```sh
['README', 'imdb.vocab', 'test', 'imdbEr.txt', 'train']
```

```python
train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)
```

```sh
['urls_neg.txt',
 'unsupBow.feat',
 'pos',
 'unsup',
 'neg',
 'urls_unsup.txt',
 'labeledBow.feat',
 'urls_pos.txt']
```

`aclImdb/train/pos` 和 `aclImdb/train/neg` 目录包含许多文本文件，每个文本文件包含一个影评。让我们挑一个看一下：

```python
sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
with open(sample_file) as f:
  print(f.read())
```

```sh
Rachel Griffiths writes and directs this award winning short film. A heartwarming story about coping with grief and cherishing the memory of those we've loved and lost. Although, only 15 minutes long, Griffiths manages to capture so much emotion and truth onto film in the short space of time. Bud Tingwell gives a touching performance as Will, a widower struggling to cope with his wife's death. Will is confronted by the harsh reality of loneliness and helplessness as he proceeds to take care of Ruth's pet cow, Tulip. The film displays the grief and responsibility one feels for those they have loved and lost. Good cinematography, great direction, and superbly acted. It will bring tears to all those who have lost a loved one, and survived.
```

### 加载数据集

接下来，从磁盘加载数据，并将其准备成适合训练的格式。

## 参考

- https://www.tensorflow.org/tutorials/keras/text_classification
