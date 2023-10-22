# 文本分类基础

- [文本分类基础](#文本分类基础)
  - [1. 简介](#1-简介)
  - [2. 情感分析](#2-情感分析)
    - [2.1 下载 IMDB 数据集](#21-下载-imdb-数据集)
    - [2.2 加载数据集](#22-加载数据集)
    - [2.3 数据预处理](#23-数据预处理)
    - [2.4 数据集性能配置](#24-数据集性能配置)
    - [2.5 创建模型](#25-创建模型)
    - [2.6 损失函数和优化器](#26-损失函数和优化器)
    - [2.7 训练模型](#27-训练模型)
    - [2.8 评估模型](#28-评估模型)
    - [2.9 趋势图](#29-趋势图)
  - [3. 导出模型](#3-导出模型)
    - [3.1 推断新数据](#31-推断新数据)
  - [4. 练习：Stack Overflow 问题的多分类](#4-练习stack-overflow-问题的多分类)
  - [5. 参考](#5-参考)

Last updated: 2022-06-20, 15:23
****

## 1. 简介

下面演示如何对文本进行分类。在 IMDB 数据集上训练一个用于情感分析（sentiment analysis））的二分类模型作为演示。

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

```txt
2.9.1
```

## 2. 情感分析

下面开始训练一个情感分析模型，即根据电影评论文本将评论分好评和差评。这是一个典型的二元分类问题。

使用的大型电影评论数据集包含来自[互联网电影数据库](https://www.imdb.com/)的 50,000 条电影评论的文本，其中 25,000 条用作训练集，25,000 条用作测试集。训练集和测试集都是平衡的，即包含相同数量的好评和差评。

### 2.1 下载 IMDB 数据集

下载并解压数据集：

```python
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset = tf.keras.utils.get_file("aclImdb_v1", url,
                                  untar=True, cache_dir=".",
                                  cache_subdir="")
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
```

```txt
Downloading data from https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
84125825/84125825 [==============================] - 22s 0us/step
```

```python
os.listdir(dataset_dir)
```

```txt
['imdb.vocab', 'imdbEr.txt', 'README', 'test', 'train']
```

```python
train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)
```

```txt
['labeledBow.feat',
 'neg',
 'pos',
 'unsup',
 'unsupBow.feat',
 'urls_neg.txt',
 'urls_pos.txt',
 'urls_unsup.txt']
```

`aclImdb/train/pos` 和 `aclImdb/train/neg` 目录包含许多文本文件，每个文本文件包含一条影评。打开一个看看：

```python
sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
with open(sample_file) as f:
    print(f.read())
```

```txt
Rachel Griffiths writes and directs this award winning short film. A heartwarming story about coping with grief and cherishing the memory of those we've loved and lost. Although, only 15 minutes long, Griffiths manages to capture so much emotion and truth onto film in the short space of time. Bud Tingwell gives a touching performance as Will, a widower struggling to cope with his wife's death. Will is confronted by the harsh reality of loneliness and helplessness as he proceeds to take care of Ruth's pet cow, Tulip. The film displays the grief and responsibility one feels for those they have loved and lost. Good cinematography, great direction, and superbly acted. It will bring tears to all those who have lost a loved one, and survived.
```

### 2.2 加载数据集

接下来，从磁盘加载数据集，并将其转换成适合训练的格式。`text_dataset_from_directory` 从目录载入数据，要求目录结构为：

```txt
main_directory/
...class_a/
......a_text_1.txt
......a_text_2.txt
...class_b/
......b_text_1.txt
......b_text_2.txt
```

对二元分类的数据集需要两个文件夹，与 `class_a` 和 `class_b` 对应。对该示例为 `aclImdb/train/pos` 和 `aclImdb/train/neg` 目录。由于 IMDB 数据集还包含其它文件夹，在使用前将其删除：

```python
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)
```

接下来，使用 `text_dataset_from_directory` 创建含标记数据集 `tf.data.Dataset`（`tf.data` 包含大量处理数据的工具）。

在进行机器学习试验时，最好将数据集分为三部分：训练集、验证集和测试集。

IMDB 数据集已经分为了训练集和测试集，还缺失验证集。下面使用 `validation_split` 参数从训练集中分出 20% 作为验证集：

```python
batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)
```

```txt
Found 25000 files belonging to 2 classes.
Using 20000 files for training.
```

可以看到，训练文件夹中有 25000 个样本，其中 20000 (80%) 用作训练。可以迭代数据集查看数据：

```python
for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(3):
        print("Review", text_batch.numpy()[i])
        print("Label", label_batch.numpy()[i])
```

```txt
Review b'"Pandemonium" is a horror movie spoof that comes off more stupid than funny. Believe me when I tell you, I love comedies. Especially comedy spoofs. "Airplane", "The Naked Gun" trilogy, "Blazing Saddles", "High Anxiety", and "Spaceballs" are some of my favorite comedies that spoof a particular genre. "Pandemonium" is not up there with those films. Most of the scenes in this movie had me sitting there in stunned silence because the movie wasn\'t all that funny. There are a few laughs in the film, but when you watch a comedy, you expect to laugh a lot more than a few times and that\'s all this film has going for it. Geez, "Scream" had more laughs than this film and that was more of a horror film. How bizarre is that?<br /><br />*1/2 (out of four)'
Label 0
Review b"David Mamet is a very interesting and a very un-equal director. His first movie 'House of Games' was the one I liked best, and it set a series of films with characters whose perspective of life changes as they get into complicated situations, and so does the perspective of the viewer.<br /><br />So is 'Homicide' which from the title tries to set the mind of the viewer to the usual crime drama. The principal characters are two cops, one Jewish and one Irish who deal with a racially charged area. The murder of an old Jewish shop owner who proves to be an ancient veteran of the Israeli Independence war triggers the Jewish identity in the mind and heart of the Jewish detective.<br /><br />This is were the flaws of the film are the more obvious. The process of awakening is theatrical and hard to believe, the group of Jewish militants is operatic, and the way the detective eventually walks to the final violent confrontation is pathetic. The end of the film itself is Mamet-like smart, but disappoints from a human emotional perspective.<br /><br />Joe Mantegna and William Macy give strong performances, but the flaws of the story are too evident to be easily compensated."
Label 0
Review b'Great documentary about the lives of NY firefighters during the worst terrorist attack of all time.. That reason alone is why this should be a must see collectors item.. What shocked me was not only the attacks, but the"High Fat Diet" and physical appearance of some of these firefighters. I think a lot of Doctors would agree with me that,in the physical shape they were in, some of these firefighters would NOT of made it to the 79th floor carrying over 60 lbs of gear. Having said that i now have a greater respect for firefighters and i realize becoming a firefighter is a life altering job. The French have a history of making great documentary\'s and that is what this is, a Great Documentary.....'
Label 1
```

需要注意的是，评论包含的是原始文本，里面包含标点符号，偶尔还有 HTML 标签，如 `<br/>`。下面会介绍如何处理。

标签为 0 或 1。要查看哪个对应好评或差评，可以通过 `Dataset` 的 `class_names` 属性查看：

```python
print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])
```

```txt
Label 0 corresponds to neg
Label 1 corresponds to pos
```

接下来创建验证集和测试集。`train\` 目录中余下的 5000 条评论用作验证集。

> [!NOTE]
> 使用 `validation_split` 和 `subset` 参数时，要么指定随机 seed，要么设置 `shuffle=False`，以确保验证集和测试集没有重叠。

```python
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)
```

```txt
Found 25000 files belonging to 2 classes.
Using 5000 files for validation.
```

```python
raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test',
    batch_size=batch_size)
```

```txt
Found 25000 files belonging to 2 classes.
```

### 2.3 数据预处理

下面使用 `tf.keras.layers.TextVectorization` 对数据进行标准化、标记化以及向量化：

- **标准化**指预处理文本，如删除标点、HTML 元素等以简化数据集；
- **标记化**指将字符串拆分为标记，如使用空格将句子拆分为单词；
- **向量化**指将标记转换为数字，以便输入神经网络。

所有这些任务都可以使用 `TextVectorization` 完成。

由于评论中包含 HTML 标签，例如 `<br />`，`TextVectorization` 的默认标准化器不会删除这些标签（默认将文本转换为小写，并去除标点符号），因此需要自定义标准化函数来删除 HTML 标签。

> [!NOTE]
> 为了避免训练集和测试集的偏差（training-testing skew），训练集和测试集应该采用相同的预处理。为了便于实现这一点，可以将 `TextVectorization` 直接包含在模型中。

```python
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')
```

开始创建 `TextVectorization` layer，并使用该 layer 执行标准化、标记化和向量化。设置 `output_mode='int'` 为每个标记创建唯一整数索引。

使用默认的 split 函数以及上面自定义标准化函数，并定义最大序列长度 `sequence_length`，该 layer 会根据该长度对序列进行填充或截断操作：

```python
max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)
```

然后调用 `adapt` 使预处理 layer 适应数据集，实现字符串到整数索引的转换。

> [!NOTE]
> 对训练集调用`adapt`，而不是测试集

创建只包含文本的数据集（不带标签），然后调用 `adapt`:

```python
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)
```

我们创建一个函数来查看该层预处理数据的效果：

```python
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label
```

从数据集中提取一个批量（32 条评论和标签）：

```python
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))
```

```txt
Review tf.Tensor(b'Great movie - especially the music - Etta James - "At Last". This speaks volumes when you have finally found that special someone.', shape=(), dtype=string)
Label neg
Vectorized review (<tf.Tensor: shape=(1, 250), dtype=int64, numpy=
array([[  86,   17,  260,    2,  222,    1,  571,   31,  229,   11, 2418,
           1,   51,   22,   25,  404,  251,   12,  306,  282,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0]], dtype=int64)>, <tf.Tensor: shape=(), dtype=int32, numpy=0>)
```

可以看到，所有的标记都已替换为整数。可以调用 `.get_vocabulary()` 查看每个标记对应的整数：

```python
print("1287 ---> ", vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> ", vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))
```

```txt
1287 --->  silent
 313 --->  night
Vocabulary size: 10000
```

数据集准备的差不多了。将上面创建的预处理步骤应用于训练集、验证集和测试集。

```python
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)
```

### 2.4 数据集性能配置

在加载数据时，有两个重要的方法可用来确保 I/O 不会阻塞：

- `.cache()` 将从磁盘加载的数据保存到内存。以确保在训练模型时数据集不会成为瓶颈。如果数据集太大而无法放入内存，使用该方法可以创建高性能的磁盘缓存，这比读取许多小文件更有效。
- `.prefetch()` 在训练时同时执行数据预处理和模型训练。

```python
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

### 2.5 创建模型

开始创建神经网络：

```python
embedding_dim = 16
```

```python
model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1)])

model.summary()
```

```txt
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, None, 16)          160016    
                                                                 
 dropout (Dropout)           (None, None, 16)          0         
                                                                 
 global_average_pooling1d (G  (None, 16)               0         
 lobalAveragePooling1D)                                          
                                                                 
 dropout_1 (Dropout)         (None, 16)                0         
                                                                 
 dense (Dense)               (None, 1)                 17        
                                                                 
=================================================================
Total params: 160,033
Trainable params: 160,033
Non-trainable params: 0
_________________________________________________________________
```

上面按顺序叠加 layer 构建分类器：

1. 第一层 `Embedding`，将评论的整数编码转换为嵌入向量。嵌入向量在模型训练时学习。输出多了一个维度，为 `(batch, sequence, embedding)`。
2. `GlobalAveragePooling1D` 对序列维度进行平均，将每个样本转换为一个固定长度的输出向量，从而可以处理变长序列。
3. 固定长度的输出向量传入一个具有 16 个隐藏单元的 `Dense` 层
4. 最后为输出 `Dense` 层。

### 2.6 损失函数和优化器

模型训练需要损失函数和优化器。由于这是一个二分类问题，并且模型输出的概率值（`sigmoimd` 激活函数），可以使用 `losses.BinaryCrossentropy` 损失函数。

为模型设置优化器和损失函数：

```python
model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
```

### 2.7 训练模型

将训练集传入 `fit` 方法：

```python
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)
```

```txt
Epoch 1/10
625/625 [==============================] - 8s 9ms/step - loss: 0.6647 - binary_accuracy: 0.6909 - val_loss: 0.6169 - val_binary_accuracy: 0.7708
Epoch 2/10
625/625 [==============================] - 4s 6ms/step - loss: 0.5512 - binary_accuracy: 0.7993 - val_loss: 0.5003 - val_binary_accuracy: 0.8214
Epoch 3/10
625/625 [==============================] - 4s 6ms/step - loss: 0.4468 - binary_accuracy: 0.8422 - val_loss: 0.4213 - val_binary_accuracy: 0.8476
Epoch 4/10
625/625 [==============================] - 4s 6ms/step - loss: 0.3793 - binary_accuracy: 0.8659 - val_loss: 0.3742 - val_binary_accuracy: 0.8598
Epoch 5/10
625/625 [==============================] - 4s 6ms/step - loss: 0.3367 - binary_accuracy: 0.8777 - val_loss: 0.3454 - val_binary_accuracy: 0.8674
Epoch 6/10
625/625 [==============================] - 4s 6ms/step - loss: 0.3055 - binary_accuracy: 0.8891 - val_loss: 0.3262 - val_binary_accuracy: 0.8714
Epoch 7/10
625/625 [==============================] - 4s 6ms/step - loss: 0.2821 - binary_accuracy: 0.8959 - val_loss: 0.3126 - val_binary_accuracy: 0.8742
Epoch 8/10
625/625 [==============================] - 4s 6ms/step - loss: 0.2625 - binary_accuracy: 0.9039 - val_loss: 0.3035 - val_binary_accuracy: 0.8746
Epoch 9/10
625/625 [==============================] - 4s 6ms/step - loss: 0.2464 - binary_accuracy: 0.9097 - val_loss: 0.2965 - val_binary_accuracy: 0.8782
Epoch 10/10
625/625 [==============================] - 4s 6ms/step - loss: 0.2313 - binary_accuracy: 0.9162 - val_loss: 0.2918 - val_binary_accuracy: 0.8786
```

### 2.8 评估模型

下面在测试集上查看模型性能。返回两个值，损失值（越小越好）和准确性：

```python
loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)
```

```txt
782/782 [==============================] - 4s 6ms/step - loss: 0.3104 - binary_accuracy: 0.8730
Loss:  0.31037768721580505
Accuracy:  0.8730000257492065
```

这个简单模型的准确率达到 87%。

### 2.9 趋势图

`model.fit()` 返回一个 `History` 对象，该对象包含训练期间的所有指标信息：

```python
history_dict = history.history
history_dict.keys()
```

```txt
dict_keys(['loss', 'binary_accuracy', 'val_loss', 'val_binary_accuracy'])
```

包含四项，即训练和验证的损失值和精度。可以使用这些信息绘图比较：

```python
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
```

![](2022-06-19-10-33-04.png)

```python
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()
```

![](2022-06-19-10-53-58.png)

上图中，点代表训练的损失和精度，实线代表验证的损失和精度。

可以看到，训练损失值随着 epoch 的增加而减少，而精度则随之增加。这与梯度下降优化的预期一致。

但是验证的损失和精度并非如此，它的准确性比训练集更早到。这是典型的过拟合，即模型在训练集上的性能比验证集上好。

对这种特殊情况，可以通过在验证精度上不再增加时停止训练，以防止过拟合。使用 `tf.keras.callbacks.EarlyStopping` callback 可以实现该功能。

## 3. 导出模型

在上例中，将文本提供给模型前应用了 `TextVectorization` layer。如果希望模型能直接处理原始字符串，可以在模型中包含 `TextVectorization` layer。为此，可以使用刚训练的权重创建新的模型：

```python
export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False),
    optimizer='adam',
    metrics=['accuracy']
)

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print(accuracy)
```

```txt
782/782 [==============================] - 8s 9ms/step - loss: 0.3104 - accuracy: 0.8730
0.8730000257492065
```

### 3.1 推断新数据

使用 `model.predict()` 推断新数据：

```python
examples = [
    "The movie was great!",
    "The movie was okay.",
    "The movie was terrible..."
]

export_model.predict(examples)
```

```txt
array([[0.6062922 ],
       [0.42909223],
       [0.34810305]], dtype=float32
```

在模型内部包括文本预处理逻辑，便于导出生产型模型，简化了部署，并减少了训练、测试的偏差。

不同位置应用 `TextVectorization`，性能有差别。如果在模型外使用 `TextVectorization`，在 GPU 上训练时，可以对数据进行异步 CPU 处理和缓冲。因此，如果在 GPU 上训练模型，则建议先在模型外使用 `TextVectorization` 以获得最佳性能，在准备部署时再将 `TextVectorization` 加入模型。

## 4. 练习：Stack Overflow 问题的多分类

本教程演示了如何在 IMDB 数据集上从头开始训练二分类模型。作为练习，可以修改本教程的模型，以预测 [Stack Overflow](https://stackoverflow.com/) 编程问题的标签。

这里已准备好一个[数据集](https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz)，其中包含 Stack Overflow 上 几千个编程问题。每个问题都有且只有一个标签，Python, CSharp, JavaScript 或 Java。任务时，将问题作为输入，预测正确的标签。

上面的数据集是从更大的公共 Stack Overflow 数据集 [BigQuery](https://console.cloud.google.com/marketplace/details/stack-exchange/stack-overflow) 中提取出来的，BigQuery 包含 1700 万个样本数据。

下载数据集后，可以看到其目录结构与之前使用的 IMDB 数据集类型：

```python
train/
...python/
......0.txt
......1.txt
...javascript/
......0.txt
......1.txt
...csharp/
......0.txt
......1.txt
...java/
......0.txt
......1.txt
```

> [!NOTE]
> 为了增加分类问题的难度，编程问题中出现的 Python, CSharp, JavaScript 和 Java 等词被替换为空格（因为许多问题都包含它们所涉及的语言）。

对该练习，可以对前面的流程做如下修改：

1. 更新下载数据集的代码，将 IMDB 替换为 Stack Overflow 数据集。由于 Stack Overflow 数据集具有类似的目录结构，因此无需做太多修改。
2. 将模型的输出层修改为 `Dense(4)`，因为现在有四个类别。
3. 编译模型时，将损失函数修改为 `tf.keras.losses.SparseCategoricalCrossentropy`。这是用于多分类问题、标签为整数的损失函数。另外，将指标改为 `metrics=['accuracy']`，因为这是多分类问题，`tf.metrics.BinaryAccuracy` 只用于二分类问题。
4. 绘图时，将 `binary_accuracy` 和 `val_binary_accuracy` 分别修改为 `accuracy` 和 `val_accuracy`。

- 导入包

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

- 下载数据集

```python
url = "https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"
dataset = tf.keras.utils.get_file("stack_overflow_16k", url,
                                  untar=True, cache_dir=".",
                                  cache_subdir="stack_overflow_16k")
dataset_dir = os.path.dirname(dataset)

os.listdir(dataset_dir)
```

```txt
['README.md', 'stack_overflow_16k.tar.gz', 'test', 'train']
```

- 查看训练集目录

```python
train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)
```

```python
['csharp', 'java', 'javascript', 'python']
```

- 查看文件

```python
sample_file = os.path.join(train_dir, 'java/666.txt')
with open(sample_file) as f:
    print(f.read())
```

```txt
"how to find max and min value so how do i find max and min value of group of numbers. ex: the numbers are ..int num[] = {1,2,3,4,5,6,7,8,9,2,2,2,2,2,};...the next things is how to find out how many times 2 appears in the array..this is what i think...char a = ""2"".int count = 0;.if (num.length = a) {.    count++;.    system.out.print (count);.}"
```

- 定义训练集

```python
batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'stack_overflow_16k/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed
)
```

```txt
Found 8000 files belonging to 4 classes.
Using 6400 files for training.
```

- 查看标签

```python
print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])
print("Label 2 corresponds to", raw_train_ds.class_names[2])
print("Label 3 corresponds to", raw_train_ds.class_names[3])
```

```txt
Label 0 corresponds to csharp
Label 1 corresponds to java
Label 2 corresponds to javascript
Label 3 corresponds to python
```

- 定义验证集

```python
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    "stack_overflow_16k/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed
)
```

```txt
Found 8000 files belonging to 4 classes.
Using 1600 files for validation.
```

- 定义测试集

```python
raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    "stack_overflow_16k/test",
    batch_size=batch_size
)
```

```txt
Found 8000 files belonging to 4 classes.
```

- 定义标准化函数

```python
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')
```

- 定义 `TextVectorization`

```python
max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
)
```

- 预处理数据集

```python
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)
```

- 性能配置

```python
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

- 创建模型

```python
embedding_dim = 16
model = keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(4)
])
model.summary()
```

```txt
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_1 (Embedding)     (None, None, 16)          160016    
                                                                 
 dropout_2 (Dropout)         (None, None, 16)          0         
                                                                 
 global_average_pooling1d_1   (None, 16)               0         
 (GlobalAveragePooling1D)                                        
                                                                 
 dropout_3 (Dropout)         (None, 16)                0         
                                                                 
 dense_1 (Dense)             (None, 4)                 68        
                                                                 
=================================================================
Total params: 160,084
Trainable params: 160,084
Non-trainable params: 0
_________________________________________________________________
```

- 编译模型

```python
model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])
```

- 训练

```python
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)
```

```txt
Epoch 1/10
200/200 [==============================] - 2s 7ms/step - loss: 1.3782 - accuracy: 0.3441 - val_loss: 1.3671 - val_accuracy: 0.5075
Epoch 2/10
200/200 [==============================] - 1s 5ms/step - loss: 1.3505 - accuracy: 0.4481 - val_loss: 1.3311 - val_accuracy: 0.5231
Epoch 3/10
200/200 [==============================] - 1s 5ms/step - loss: 1.3018 - accuracy: 0.5361 - val_loss: 1.2741 - val_accuracy: 0.5838
Epoch 4/10
200/200 [==============================] - 1s 5ms/step - loss: 1.2356 - accuracy: 0.5948 - val_loss: 1.2012 - val_accuracy: 0.6200
Epoch 5/10
200/200 [==============================] - 1s 6ms/step - loss: 1.1576 - accuracy: 0.6391 - val_loss: 1.1241 - val_accuracy: 0.6612
Epoch 6/10
200/200 [==============================] - 1s 6ms/step - loss: 1.0803 - accuracy: 0.6778 - val_loss: 1.0519 - val_accuracy: 0.6963
Epoch 7/10
200/200 [==============================] - 1s 5ms/step - loss: 1.0098 - accuracy: 0.7069 - val_loss: 0.9877 - val_accuracy: 0.7219
Epoch 8/10
200/200 [==============================] - 1s 6ms/step - loss: 0.9463 - accuracy: 0.7248 - val_loss: 0.9314 - val_accuracy: 0.7287
Epoch 9/10
200/200 [==============================] - 1s 6ms/step - loss: 0.8909 - accuracy: 0.7459 - val_loss: 0.8827 - val_accuracy: 0.7412
Epoch 10/10
200/200 [==============================] - 1s 5ms/step - loss: 0.8418 - accuracy: 0.7597 - val_loss: 0.8410 - val_accuracy: 0.7556
```

- 验证模型

```python
loss, accuracy = model.evaluate(test_ds)
print("Loss:", loss)
print("Accuracy:", accuracy)
```

```txt
250/250 [==============================] - 1s 5ms/step - loss: 0.8742 - accuracy: 0.7249
Loss: 0.8742028474807739
Accuracy: 0.7248749732971191
```

- 提取训练数据

```python
history_dict = history.history
history_dict.keys()
```

```txt
dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
```

- 绘制损失图

```python
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss, 'bo', label='Trainnig loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()
```

![](2022-06-20-16-01-00.png)

- 绘制精度图

```python
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()
```

![](2022-06-20-16-01-26.png)

## 5. 参考

- https://www.tensorflow.org/tutorials/keras/text_classification
