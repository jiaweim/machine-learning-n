# 基于 RNN 的文本分类

- [基于 RNN 的文本分类](#基于-rnn-的文本分类)
  - [简介](#简介)
  - [输入管道](#输入管道)
  - [创建 text encoder](#创建-text-encoder)
  - [参考](#参考)

2022-01-07, 15:19
***

## 简介

下面在 IMDB 电影评论数据集上训练一个情感分析 RNN 网络。首先，导入必要的包：

```python
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

tfds.disable_progress_bar()
```

导入 matplotlib 并创建辅助绘图函数：

```python
import matplotlib.pyplot as plt

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])
```

## 输入管道

IMDB 数据集是一个二元分类数据集，所有的评论只有好评和差评两类。

使用 TFDS 下载数据集：

```python
dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

train_dataset.element_spec
```

```txt
(TensorSpec(shape=(), dtype=tf.string, name=None),
 TensorSpec(shape=(), dtype=tf.int64, name=None))
```

返回的为 (text, label) 对：

```python
for example, label in train_dataset.take(1):
    print('text: ', example.numpy())
    print('label: ', label.numpy())
```

```txt
text:  b"This was an absolutely terrible movie. Don't be lured in by Christopher Walken or Michael Ironside. Both are great actors, but this must simply be their worst role in history. Even their great acting could not redeem this movie's ridiculous storyline. This movie is an early nineties US propaganda piece. The most pathetic scenes were those when the Columbian rebels were making their cases for revolutions. Maria Conchita Alonso appeared phony, and her pseudo-love affair with Walken was nothing but a pathetic emotional plug in a movie that was devoid of any real meaning. I am disappointed that there are movies like this, ruining actor's like Christopher Walken's good name. I could barely sit through it."
label:  0
```

训练数据乱序，并创建 `(text, label)` batch：

```python
BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
```

```python
for example, label in train_dataset.take(1):
    print('texts: ', example.numpy()[:3])
    print()
    print('labels: ', label.numpy()[:3])
```

```txt
texts:  [b"This game is amazing. Really, you should get it if you don't have it. Although it is ancient now it was amazing when it came out. I believe that this game will always be a classic. It's just as good a Super Mario World or so. When I was young, my friend and I would sit and play this game for hours trying to beat it which we eventually did. It's not nearly as advanced as Super Mario Galaxy, but if you are a fellow Mario fan it is essential. It's fun entertaining and challenging. Everything you could want out of a fantasy game except for good graphics, (well it did come out in 1996.) ROCK ON 4EVA MARIO LUIGI AND YOSHI!!! Nintendo is the best!"
 b"This inferior sequel based by the characters created by David Selzer and Harvey Bernhard(also producer) concern on a matrimony named Gene(Michael Woods) and Karen York(Faye Grant). They adopt a little girl named Delia from a convent. Gene York about re-elect for congressman and he presides the financing committee. Meanwhile, Delia seems to be around when inexplicable deaths happen. She creates wreak havoc when goes a metaphysical fair, as stores of numerology, therapy, counselling heal,yoga, tarots, among others are destroyed. Karen York hires an eye private(Michael Lerner) to investigate the weird and bizarre events.<br /><br />This TV sequel displays thrills, chills, creepy events and gory killing. Delia such as Damien seems to dispatch new eerie murder every few minutes of film, happening horrible killings . The chief excitement lies in watching what new and innocent victim can be made by the middling special effects. Furthermore, mediocre protagonists, Faye Grant and Michael Woods, however nice cast secondary, such as Michael Lerner,Madison Mason, Duncan Fraser and the recently deceased Don S Davis, he was an Army captain turned into acting. As always , excellent musical score taken from Omen I and III by the great Jerry Goldsmith. The movie is exclusively for hardcore followers Omen saga. The motion picture is badly directed by Jorge Montesi and Dominique Othenin Girard. Previous and much better versions are the following : The immensely superior original 'Omen'(Gregory Peck, Lee Remick)by Richard Donner; 'Damien'(William Holden, Lee Grant) by Don Taylor; 'Final conflict'(Sam Neil and Tisa Harrow) by Grahame Baker. Rating : Below average."
 b"This is my first comment on IMDb website, and the reason I'm writing it is that we're talking about ONE OF THE BEST FILMS EVER! 'Ne goryuy!' will make you laugh and cry at the same time, you will fall in love (if you're not a fan yet!) with Georgian choir singing tradition, and possibly you will accept the hardships of your own existence and just feel good after watching it:) What I like a lot about this film is that actors in the non-leading roles create vivid and memorable characters and are just as interesting and important as the central character. The film is starring Vahtang Kikabidze (who is great), but you will remember every single face around him in the film. You will find yourself quoting their lines, that have become household names for so many Russian-speaking people. A film to live with. Simple, yet deep, you will want to watch it again and again."]

labels:  [1 0 1]
```

## 创建 text encoder

tfds 加载的原始文本需要进行处理才能用于训练模型。最简单的方式是使用 `TextVectorization` layer。该 layer 有很多功能，不过下面只使用默认配置。

创建 `TextVectorization` layer，将数据集文本传入 `.adapt` 方法：

```python
VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))
```

`.adapt` 方法设置 layer 的 vocab。下面是前 20 个 tokens。在 padding 和 unknown 之后，它们按频率进行排序：

```python
vocab = np.array(encoder.get_vocabulary())
vocab[:20]
```

```sh
array(['', '[UNK]', 'the', 'and', 'a', 'of', 'to', 'is', 'in', 'it', 'i',
       'this', 'that', 'br', 'was', 'as', 'for', 'with', 'movie', 'but'],
      dtype='<U14')
```




## 参考

- https://www.tensorflow.org/text/tutorials/text_classification_rnn
- https://tensorflow.google.cn/text/tutorials/text_classification_rnn
