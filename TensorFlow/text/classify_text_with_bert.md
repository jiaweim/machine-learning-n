# 用 BERT 做文本分类

- [用 BERT 做文本分类](#用-bert-做文本分类)
  - [简介](#简介)
  - [BERT](#bert)
  - [设置](#设置)
  - [情感分析](#情感分析)
    - [下载 IMDB 数据集](#下载-imdb-数据集)
  - [从 TensorFlow Hub 加载模型](#从-tensorflow-hub-加载模型)
  - [参考](#参考)

***

## 简介

本教程包含 fine-tune BERT 模型的完整代码，用来对 IMDB 数据集做情感分析。除了训练模型外，还包括将文本预处理为合适格式。包括：

- 加载 IMDB 数据集
- 从 TensorFlow Hub 加载 BERT 模型
- 结合 BERT 和分类器来构建自己的模型
- 训练自己的模型，微调 BERT
- 保存模型，用来分类句子

## BERT

BERT 和其它基于 Transformer encoder 的架构在 NLP 的各种任务中取得了巨大成功。Encoder 输出的自然语言向量空间表示适合在深度学习模型中使用。BERT 系列模型使用 Transformer encoder 架构，使用 token 前后的所有 token 预测该 token，因此称为 Bidirectional Encoder Representations from Transformers (BERT)。

BERT 模型通常在大量文本语料库上进行预训练，然后针对特定任务进行微调（fine-tune）。

## 设置

安装 tensorflow-text 用于预处理输入：

```powershell
pip install -q -U "tensorflow-text==2.8.*"
```

安装 tf-models-official，因为要使用 AdamW optimizer:

```powershell
pip install -q tf-models-official==2.7.0
```

导入所需包：

```python
import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')
```

## 情感分析

下面训练一个情感分析模型，根据评论文本将影评分为*好评*和*差评*。

将使用来自 [Internet Movie Database](https://www.imdb.com/) 包含 50,000 影评的 [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)。

### 下载 IMDB 数据集

下载并解压 IMDB 数据集，并查看其目录结构：

```python
url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

dataset = tf.keras.utils.get_file('aclImdb_v1.tar.gz', url,
                                  untar=True, cache_dir='.',
                                  cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

train_dir = os.path.join(dataset_dir, 'train')

# remove unused folders to make it easier to load the data
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)
```

```txt
Downloading data from https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
84131840/84125825 [==============================] - 6s 0us/step
84140032/84125825 [==============================] - 6s 0us/step
```

接下来，使用 `text_dataset_from_directory` 创建 `tf.data.Dataset`。

IMDB 数据集已经划分为训练集和测试集，还缺少验证集。下面使用 `validation_split` 参数从训练集中以 80:20 分割一部分作为验证集。

> **NOTE**: 使用 `validation_split` 和 `subset` 参数时，必须指定 random 种子，或设置 `shuffle=False`，否则训练集和验证集会有重叠。

```python
AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

class_names = raw_train_ds.class_names
train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)

val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

test_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test',
    batch_size=batch_size)

test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

```txt
Found 25000 files belonging to 2 classes.
Using 20000 files for training.
Found 25000 files belonging to 2 classes.
Using 5000 files for validation.
Found 25000 files belonging to 2 classes.
```

查看评论：

```python
for text_batch, label_batch in train_ds.take(1):
  for i in range(3):
    print(f'Review: {text_batch.numpy()[i]}')
    label = label_batch.numpy()[i]
    print(f'Label : {label} ({class_names[label]})')
```

```txt
Review: b'"Pandemonium" is a horror movie spoof that comes off more stupid than funny. Believe me when I tell you, I love comedies. Especially comedy spoofs. "Airplane", "The Naked Gun" trilogy, "Blazing Saddles", "High Anxiety", and "Spaceballs" are some of my favorite comedies that spoof a particular genre. "Pandemonium" is not up there with those films. Most of the scenes in this movie had me sitting there in stunned silence because the movie wasn\'t all that funny. There are a few laughs in the film, but when you watch a comedy, you expect to laugh a lot more than a few times and that\'s all this film has going for it. Geez, "Scream" had more laughs than this film and that was more of a horror film. How bizarre is that?<br /><br />*1/2 (out of four)'
Label : 0 (neg)
Review: b"David Mamet is a very interesting and a very un-equal director. His first movie 'House of Games' was the one I liked best, and it set a series of films with characters whose perspective of life changes as they get into complicated situations, and so does the perspective of the viewer.<br /><br />So is 'Homicide' which from the title tries to set the mind of the viewer to the usual crime drama. The principal characters are two cops, one Jewish and one Irish who deal with a racially charged area. The murder of an old Jewish shop owner who proves to be an ancient veteran of the Israeli Independence war triggers the Jewish identity in the mind and heart of the Jewish detective.<br /><br />This is were the flaws of the film are the more obvious. The process of awakening is theatrical and hard to believe, the group of Jewish militants is operatic, and the way the detective eventually walks to the final violent confrontation is pathetic. The end of the film itself is Mamet-like smart, but disappoints from a human emotional perspective.<br /><br />Joe Mantegna and William Macy give strong performances, but the flaws of the story are too evident to be easily compensated."
Label : 0 (neg)
Review: b'Great documentary about the lives of NY firefighters during the worst terrorist attack of all time.. That reason alone is why this should be a must see collectors item.. What shocked me was not only the attacks, but the"High Fat Diet" and physical appearance of some of these firefighters. I think a lot of Doctors would agree with me that,in the physical shape they were in, some of these firefighters would NOT of made it to the 79th floor carrying over 60 lbs of gear. Having said that i now have a greater respect for firefighters and i realize becoming a firefighter is a life altering job. The French have a history of making great documentary\'s and that is what this is, a Great Documentary.....'
Label : 1 (pos)
2022-03-29 12:30:15.775528: W tensorflow/core/kernels/data/cache_dataset_ops.cc:768] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.
```

## 从 TensorFlow Hub 加载模型

在这里，可以选择从 TensorFlow Hub 加载哪个 BERT 模型进行微调。有多种 BERT 模型可供选择：

- 由 BERT 作者发布的 [BERT-Base](https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3), [Uncased](https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3) 和[其它 7 个](https://tfhub.dev/google/collections/bert/1)模型。
- [Small BERTs](https://tfhub.dev/google/collections/bert/1)具有相同的总体架构，但是 Transformer block 更少或更小，是你能够在速度、大小和质量间进行权衡。
- ALBERT：4 种不同的 "A Lite BERT"，通过在不同层之间共享参数减少模型尺寸（不减少计算时间）。
- [BERT Experts](https://tfhub.dev/google/collections/experts/bert/1) ：8 个模型 BERT-base 架构模型，但提供了不同的 pre-training 领域，以更紧密地与目标任务一致。
- [Electra](https://tfhub.dev/google/collections/electra/1) 

## 参考

- https://www.tensorflow.org/text/tutorials/classify_text_with_bert
