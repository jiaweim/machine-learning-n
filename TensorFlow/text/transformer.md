# Transformer

- [Transformer](#transformer)
  - [简介](#简介)
  - [设置](#设置)
  - [下载数据集](#下载数据集)
  - [文本标记化和去标记化](#文本标记化和去标记化)
  - [参考](#参考)

***

## 简介

本教程训练一个 Transformer 模型，用来翻译[葡萄牙语-英语](https://www.tensorflow.org/datasets/catalog/ted_hrlr_translate#ted_hrlr_translatept_to_en)数据集。演示如何使用 TensorFlow 低级 API 和 Keras 函数从头开始构建 Transformer 模型。如果使用 [tf.keras.layers.MultiHeadAttention](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention) 之类的内置 API，则实现起来更简单。

Transformer 模型的核心思想是 self-attention，即关注输入序列不同位置以计算该序列表示（representation）的能力。Transformer 创建 self-attention layers 堆栈，在下面会分布讲解。

Transformer 模型使用 self-attention layers 处理可变大小输入，而不是经典的 RNN 或 CNN。这种通用架构有许多优点：

- 不假设数据之间的时间/空间关系，是处理一组对象的理想选择
- layer 输出可以并行计算，而不像 RNN 的串行计算
- 距离很远的元素可以直接影响彼此的输出，而无需经过许多 RNN 时间步或卷积层，可以参考 [Scene Memory Transformer](https://arxiv.org/pdf/1903.03878.pdf)
- 可以学习远程依赖，这在许多序列任务中都是挑战。

该架构的缺点有：

- 对时间序列，时间步的输出是从整个历史中计算，而不仅仅是输入和当前 hidden-state，这可能比较低效；
- 如果输入确实具有时间/空间关系，如文本，则必须添加位置编码，否则模型只能看到 bag of words

根据本教程训练模型后，能够将葡萄牙语翻译为英语。

![](images/2022-08-06-23-03-27.png)

## 设置

- 导入包

```python
import logging
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf

import tensorflow_text
```

- 设置日志

```python
logging.getLogger("tensorflow").setLevel(logging.ERROR)  # suppress warnings
```

## 下载数据集

使用 Tensorflow dataset 加载来自 [TED 演讲开放翻译项目](https://www.ted.com/participate/translate)的[葡萄牙语-英语](https://github.com/neulab/word-embeddings-for-nmt)翻译数据集。

该数据集包含大约 50000 训练样本，1100 验证样本和 2000 测试样本。

```python
examples, metadata = tfds.load(
    "ted_hrlr_translate/pt_to_en", with_info=True, as_supervised=True
)
train_examples, val_examples = examples["train"], examples["validation"]
```

返回的 `tf.data.Dataset` 包含成对的文本样本：

```python
# 取 3 个样本，查看葡萄牙语和英语
for pt_examples, en_examples in train_examples.batch(3).take(1):
    for pt in pt_examples.numpy():
        print(pt.decode("utf-8"))

    print()

    for en in en_examples.numpy():
        print(en.decode("utf-8"))
```

```txt
e quando melhoramos a procura , tiramos a única vantagem da impressão , que é a serendipidade .
mas e se estes fatores fossem ativos ?
mas eles não tinham a curiosidade de me testar .

and when you improve searchability , you actually take away the one advantage of print , which is serendipity .
but what if it were active ?
but they did n't test for curiosity .
```

## 文本标记化和去标记化

不能直接使用文本训练模型，需要先将文本转换为数字表示。通常将文本转换为 token ID 序列，然后将 token ID 用作嵌入的输入。



## 参考

- https://www.tensorflow.org/text/tutorials/transformer
