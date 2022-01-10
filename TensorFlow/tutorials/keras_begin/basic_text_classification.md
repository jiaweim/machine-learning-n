# 基本文本分类

- [基本文本分类](#基本文本分类)
  - [简介](#简介)
  - [情感分析](#情感分析)
    - [下载 IMDB 数据集](#下载-imdb-数据集)
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
2.7.0
```

## 情感分析

下面训练一个情感分析模型，根据电影的评论文本将评论分好评或差评。

下面使用大型电影评论数据集，该数据集包含来自互联网电影数据库的 50,000 个电影评论的文本。这些评估分为 25,000 个训练样本和 25,000 个测试样本。训练样本和测试样本都是平衡的，即包含相同数量的好评和差评。

### 下载 IMDB 数据集

下载并解压数据集：

```python

```

## 参考

- https://www.tensorflow.org/tutorials/keras/text_classification
