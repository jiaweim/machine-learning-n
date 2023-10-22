# 基于 TensorFlow Hub 的文本分类：影评

- [基于 TensorFlow Hub 的文本分类：影评](#基于-tensorflow-hub-的文本分类影评)
  - [1. 简介](#1-简介)
  - [2. 下载 IMDB 数据集](#2-下载-imdb-数据集)
  - [3. 查看数据](#3-查看数据)
  - [4. 构建模型](#4-构建模型)
    - [4.1 损失函数和优化器](#41-损失函数和优化器)
  - [5. 训练模型](#5-训练模型)
  - [6. 评估模型](#6-评估模型)
  - [7. 参考](#7-参考)

Last updated: 2022-06-20, 17:14
@author Jiawei Mao
****

## 1. 简介

下面实现一个将电影评论分类为好评和差评的模型。这是一个典型的二元分类模型，在机器学习中有着广泛的应用。

使用的大型电影评论数据集包含来自互联网电影数据库的 50,000 条电影评论的文本，其中 25,000 条用作训练集，25,000 条用作测试集。训练集和测试集都是平衡的，即包含相同数量的好评和差评。

本教程主要演示使用 [TensorFlow Hub](https://tfhub.dev/) 和 Keras 实现迁移学习。使用 `tf.keras` 高级 API 构建和训练模型，使用 `tensorflow_hub` 从 TFHub 加载已训练模型。

```powershell
pip install tensorflow-hub
pip install tensorflow-datasets
```

```python
import os
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")
```

```txt
Version:  2.9.1
Eager mode:  True
Hub version:  0.12.0
GPU is available
```

## 2. 下载 IMDB 数据集

在 [imdb reviews](http://ai.stanford.edu/~amaas/data/sentiment/) 和 [TensorFlow datasets](https://www.tensorflow.org/datasets) 都有 IMDB 数据集。下载 IMDB 数据集：

```python
# 训练集按 6:4 拆分，15,000 用作训练，10,000 用作验证
# 测试集包含 25,000 样本
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)
```

## 3. 查看数据

先花点时间来理解数据格式。数据集的每个样本包含影评及其标签。影评没有经过任何预处理。标签为 0（差评） 或 1（好评）。

查看前 10 个样本：

```python
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
train_examples_batch
```

```txt
<tf.Tensor: shape=(10,), dtype=string, numpy=
array([b"This was an absolutely terrible movie. Don't be lured in by Christopher Walken or Michael Ironside. Both are great actors, but this must simply be their worst role in history. Even their great acting could not redeem this movie's ridiculous storyline. This movie is an early nineties US propaganda piece. The most pathetic scenes were those when the Columbian rebels were making their cases for revolutions. Maria Conchita Alonso appeared phony, and her pseudo-love affair with Walken was nothing but a pathetic emotional plug in a movie that was devoid of any real meaning. I am disappointed that there are movies like this, ruining actor's like Christopher Walken's good name. I could barely sit through it.",
       b'I have been known to fall asleep during films, but this is usually due to a combination of things including, really tired, being warm and comfortable on the sette and having just eaten a lot. However on this occasion I fell asleep because the film was rubbish. The plot development was constant. Constantly slow and boring. Things seemed to happen, but with no explanation of what was causing them or why. I admit, I may have missed part of the film, but i watched the majority of it and everything just seemed to happen of its own accord without any real concern for anything else. I cant recommend this film at all.',
       b'Mann photographs the Alberta Rocky Mountains in a superb fashion, and Jimmy Stewart and Walter Brennan give enjoyable performances as they always seem to do. <br /><br />But come on Hollywood - a Mountie telling the people of Dawson City, Yukon to elect themselves a marshal (yes a marshal!) and to enforce the law themselves, then gunfighters battling it out on the streets for control of the town? <br /><br />Nothing even remotely resembling that happened on the Canadian side of the border during the Klondike gold rush. Mr. Mann and company appear to have mistaken Dawson City for Deadwood, the Canadian North for the American Wild West.<br /><br />Canadian viewers be prepared for a Reefer Madness type of enjoyable howl with this ludicrous plot, or, to shake your head in disgust.',
       b'This is the kind of film for a snowy Sunday afternoon when the rest of the world can go ahead with its own business as you descend into a big arm-chair and mellow for a couple of hours. Wonderful performances from Cher and Nicolas Cage (as always) gently row the plot along. There are no rapids to cross, no dangerous waters, just a warm and witty paddle through New York life at its best. A family film in every sense and one that deserves the praise it received.',
       b'As others have mentioned, all the women that go nude in this film are mostly absolutely gorgeous. The plot very ably shows the hypocrisy of the female libido. When men are around they want to be pursued, but when no "men" are around, they become the pursuers of a 14 year old boy. And the boy becomes a man really fast (we should all be so lucky at this age!). He then gets up the courage to pursue his true love.',
       b"This is a film which should be seen by anybody interested in, effected by, or suffering from an eating disorder. It is an amazingly accurate and sensitive portrayal of bulimia in a teenage girl, its causes and its symptoms. The girl is played by one of the most brilliant young actresses working in cinema today, Alison Lohman, who was later so spectacular in 'Where the Truth Lies'. I would recommend that this film be shown in all schools, as you will never see a better on this subject. Alison Lohman is absolutely outstanding, and one marvels at her ability to convey the anguish of a girl suffering from this compulsive disorder. If barometers tell us the air pressure, Alison Lohman tells us the emotional pressure with the same degree of accuracy. Her emotional range is so precise, each scene could be measured microscopically for its gradations of trauma, on a scale of rising hysteria and desperation which reaches unbearable intensity. Mare Winningham is the perfect choice to play her mother, and does so with immense sympathy and a range of emotions just as finely tuned as Lohman's. Together, they make a pair of sensitive emotional oscillators vibrating in resonance with one another. This film is really an astonishing achievement, and director Katt Shea should be proud of it. The only reason for not seeing it is if you are not interested in people. But even if you like nature films best, this is after all animal behaviour at the sharp edge. Bulimia is an extreme version of how a tormented soul can destroy her own body in a frenzy of despair. And if we don't sympathise with people suffering from the depths of despair, then we are dead inside.",
       b'Okay, you have:<br /><br />Penelope Keith as Miss Herringbone-Tweed, B.B.E. (Backbone of England.) She\'s killed off in the first scene - that\'s right, folks; this show has no backbone!<br /><br />Peter O\'Toole as Ol\' Colonel Cricket from The First War and now the emblazered Lord of the Manor.<br /><br />Joanna Lumley as the ensweatered Lady of the Manor, 20 years younger than the colonel and 20 years past her own prime but still glamourous (Brit spelling, not mine) enough to have a toy-boy on the side. It\'s alright, they have Col. Cricket\'s full knowledge and consent (they guy even comes \'round for Christmas!) Still, she\'s considerate of the colonel enough to have said toy-boy her own age (what a gal!)<br /><br />David McCallum as said toy-boy, equally as pointlessly glamourous as his squeeze. Pilcher couldn\'t come up with any cover for him within the story, so she gave him a hush-hush job at the Circus.<br /><br />and finally:<br /><br />Susan Hampshire as Miss Polonia Teacups, Venerable Headmistress of the Venerable Girls\' Boarding-School, serving tea in her office with a dash of deep, poignant advice for life in the outside world just before graduation. Her best bit of advice: "I\'ve only been to Nancherrow (the local Stately Home of England) once. I thought it was very beautiful but, somehow, not part of the real world." Well, we can\'t say they didn\'t warn us.<br /><br />Ah, Susan - time was, your character would have been running the whole show. They don\'t write \'em like that any more. Our loss, not yours.<br /><br />So - with a cast and setting like this, you have the re-makings of "Brideshead Revisited," right?<br /><br />Wrong! They took these 1-dimensional supporting roles because they paid so well. After all, acting is one of the oldest temp-jobs there is (YOU name another!)<br /><br />First warning sign: lots and lots of backlighting. They get around it by shooting outdoors - "hey, it\'s just the sunlight!"<br /><br />Second warning sign: Leading Lady cries a lot. When not crying, her eyes are moist. That\'s the law of romance novels: Leading Lady is "dewy-eyed."<br /><br />Henceforth, Leading Lady shall be known as L.L.<br /><br />Third warning sign: L.L. actually has stars in her eyes when she\'s in love. Still, I\'ll give Emily Mortimer an award just for having to act with that spotlight in her eyes (I wonder . did they use contacts?)<br /><br />And lastly, fourth warning sign: no on-screen female character is "Mrs." She\'s either "Miss" or "Lady."<br /><br />When all was said and done, I still couldn\'t tell you who was pursuing whom and why. I couldn\'t even tell you what was said and done.<br /><br />To sum up: they all live through World War II without anything happening to them at all.<br /><br />OK, at the end, L.L. finds she\'s lost her parents to the Japanese prison camps and baby sis comes home catatonic. Meanwhile (there\'s always a "meanwhile,") some young guy L.L. had a crush on (when, I don\'t know) comes home from some wartime tough spot and is found living on the street by Lady of the Manor (must be some street if SHE\'s going to find him there.) Both war casualties are whisked away to recover at Nancherrow (SOMEBODY has to be "whisked away" SOMEWHERE in these romance stories!)<br /><br />Great drama.',
       b'The film is based on a genuine 1950s novel.<br /><br />Journalist Colin McInnes wrote a set of three "London novels": "Absolute Beginners", "City of Spades" and "Mr Love and Justice". I have read all three. The first two are excellent. The last, perhaps an experiment that did not come off. But McInnes\'s work is highly acclaimed; and rightly so. This musical is the novelist\'s ultimate nightmare - to see the fruits of one\'s mind being turned into a glitzy, badly-acted, soporific one-dimensional apology of a film that says it captures the spirit of 1950s London, and does nothing of the sort.<br /><br />Thank goodness Colin McInnes wasn\'t alive to witness it.',
       b'I really love the sexy action and sci-fi films of the sixties and its because of the actress\'s that appeared in them. They found the sexiest women to be in these films and it didn\'t matter if they could act (Remember "Candy"?). The reason I was disappointed by this film was because it wasn\'t nostalgic enough. The story here has a European sci-fi film called "Dragonfly" being made and the director is fired. So the producers decide to let a young aspiring filmmaker (Jeremy Davies) to complete the picture. They\'re is one real beautiful woman in the film who plays Dragonfly but she\'s barely in it. Film is written and directed by Roman Coppola who uses some of his fathers exploits from his early days and puts it into the script. I wish the film could have been an homage to those early films. They could have lots of cameos by actors who appeared in them. There is one actor in this film who was popular from the sixties and its John Phillip Law (Barbarella). Gerard Depardieu, Giancarlo Giannini and Dean Stockwell appear as well. I guess I\'m going to have to continue waiting for a director to make a good homage to the films of the sixties. If any are reading this, "Make it as sexy as you can"! I\'ll be waiting!',
       b'Sure, this one isn\'t really a blockbuster, nor does it target such a position. "Dieter" is the first name of a quite popular German musician, who is either loved or hated for his kind of acting and thats exactly what this movie is about. It is based on the autobiography "Dieter Bohlen" wrote a few years ago but isn\'t meant to be accurate on that. The movie is filled with some sexual offensive content (at least for American standard) which is either amusing (not for the other "actors" of course) or dumb - it depends on your individual kind of humor or on you being a "Bohlen"-Fan or not. Technically speaking there isn\'t much to criticize. Speaking of me I find this movie to be an OK-movie.'],
      dtype=object)>
```

查看前 10 个标签：

```python
train_labels_batch
```

```txt
<tf.Tensor: shape=(10,), dtype=int64, numpy=array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0], dtype=int64)>
```

## 4. 构建模型

`tf.keras` API 通过堆叠 layer 构建神经网络模型，有三个重要架构需要考虑：

- 如何表示文本？
- 在模型中使用多少层？
- 每层使用多少隐藏单元？

对本例，输入数据由句子文本组成，标签为 0 或 1。表示文本的一种方法是将句子转换为嵌入向量。将预训练的文本嵌入作为第一层，至少有三个优点：

- 不需要操心文本预处理；
- 受益于迁移学习；
- 嵌入大小固定，处理更简单。

对本例，使用 TensorFlow Hub 提供的一个预先训练好的文本嵌入模型 [google/nnlm-en-dim50/2](https://tfhub.dev/google/nnlm-en-dim50/2) 。

> [!TIP]
> 如果打不开 https://tfhub.dev，可以访问镜像网站 https://hub.tensorflow.google.cn/

TFHub 中还有许多预训练好的文本嵌入，均可以在本教程中使用：

- [nnlm-en-dim128/2](https://tfhub.dev/google/nnlm-en-dim128/2) - 与 `google/nnlm-en-dim50/2` 使用相同的 NNLM 架构、相同的数据集，但是嵌入维度更大。更大的嵌入维度可以改进任务，但是需要更长的训练时间。
- [nnlm-en-dim128-with-normalization/2](https://tfhub.dev/google/nnlm-en-dim128-with-normalization/2) - 与 `google/nnlm-en-dim128/2` 一样，但是额外添加了文本标准化，如删除标点符号。如果文本中包含标点符号或其它字符，会有所帮助。
- [google/universal-sentence-encoder/4](https://tfhub.dev/google/universal-sentence-encoder/4) - 使用深度平均网络（deep averaging network, DAN）encoder 训练的一个更大的模型，嵌入维度 512.

在 TFHub 上还有许多文本嵌入模型。

首先，创建一个 Keras layer，使用 TFHub 模型来嵌入文本。注意，不管输入文本的长度是多少，嵌入的输出 shape 都是 `(num_examples, embedding_dimension)`：

```python
embedding = "https://hub.tensorflow.google.cn/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])
```

```txt
<tf.Tensor: shape=(3, 50), dtype=float32, numpy=
array([[ 0.5423195 , -0.0119017 ,  0.06337538,  0.06862972, -0.16776837,
        -0.10581174,  0.16865303, -0.04998824, -0.31148055,  0.07910346,
         0.15442263,  0.01488662,  0.03930153,  0.19772711, -0.12215476,
        -0.04120981, -0.2704109 , -0.21922152,  0.26517662, -0.80739075,
         0.25833532, -0.3100421 ,  0.28683215,  0.1943387 , -0.29036492,
         0.03862849, -0.7844411 , -0.0479324 ,  0.4110299 , -0.36388892,
        -0.58034706,  0.30269456,  0.3630897 , -0.15227164, -0.44391504,
         0.19462997,  0.19528408,  0.05666234,  0.2890704 , -0.28468323,
        -0.00531206,  0.0571938 , -0.3201318 , -0.04418665, -0.08550783,
        -0.55847436, -0.23336391, -0.20782952, -0.03543064, -0.17533456],
       [ 0.56338924, -0.12339553, -0.10862679,  0.7753425 , -0.07667089,
        -0.15752277,  0.01872335, -0.08169781, -0.3521876 ,  0.4637341 ,
        -0.08492756,  0.07166859, -0.00670817,  0.12686075, -0.19326553,
        -0.52626437, -0.3295823 ,  0.14394785,  0.09043556, -0.5417555 ,
         0.02468163, -0.15456742,  0.68333143,  0.09068331, -0.45327246,
         0.23180096, -0.8615696 ,  0.34480393,  0.12838456, -0.58759046,
        -0.4071231 ,  0.23061076,  0.48426893, -0.27128142, -0.5380916 ,
         0.47016326,  0.22572741, -0.00830663,  0.2846242 , -0.304985  ,
         0.04400365,  0.25025874,  0.14867121,  0.40717036, -0.15422426,
        -0.06878027, -0.40825695, -0.3149215 ,  0.09283665, -0.20183425],
       [ 0.7456154 ,  0.21256861,  0.14400336,  0.5233862 ,  0.11032254,
         0.00902788, -0.3667802 , -0.08938274, -0.24165542,  0.33384594,
        -0.11194605, -0.01460047, -0.0071645 ,  0.19562712,  0.00685216,
        -0.24886718, -0.42796347,  0.18620004, -0.05241098, -0.66462487,
         0.13449019, -0.22205497,  0.08633006,  0.43685386,  0.2972681 ,
         0.36140734, -0.7196889 ,  0.05291241, -0.14316116, -0.1573394 ,
        -0.15056328, -0.05988009, -0.08178931, -0.15569411, -0.09303783,
        -0.18971172,  0.07620788, -0.02541647, -0.27134508, -0.3392682 ,
        -0.10296468, -0.27275252, -0.34078008,  0.20083304, -0.26644835,
         0.00655449, -0.05141488, -0.04261917, -0.45413622,  0.20023568]],
      dtype=float32)>
```

构建完整模型：

```python
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.summary()
```

```txt
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 keras_layer (KerasLayer)    (None, 50)                48190600  
                                                                 
 dense (Dense)               (None, 16)                816       
                                                                 
 dense_1 (Dense)             (None, 1)                 17        
                                                                 
=================================================================
Total params: 48,191,433
Trainable params: 48,191,433
Non-trainable params: 0
_________________________________________________________________
```

这些 layers 按顺序堆叠构建分类器：

1. TFHub layer，该层使用预训练的模型将句子映射到嵌入向量。使用的预训练文本嵌入模型（google/nnlm-en-dim50/2）将文本拆分为标记，嵌入每个标记，然后合并嵌入。最终的 shape 为 `(num_examples, embedding_dimension)`。对该 NNLM 模型，`embedding_dimension` 为 50.
2. TFHub layer 输出的定长向量输入包含 16 个隐藏单元的 `Dense` 层
3. 最后一层是包含 1 个隐藏单元的 `Dense` 层

### 4.1 损失函数和优化器

模型训练需要损失函数和优化器。对二元分类问题，并且模型输出 logits (具有线性激活函数的 single-unit layer)，使用 `binary_crossentropy` 损失函数。当然也可以用 `mean_squared_error`，但是 `binary_crossentropy` 更适合处理概率，它计算不同概率分布之间的距离，对本例，它计算真实分布和预测之间的距离。

设置优化器和损失函数：

```python
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

## 5. 训练模型

设置 batch 为 512，训练 10 个 epochs，即对 `x_train` 和 `y_train` 张量的所有样本迭代 10 次。训练时，使用验证集监控损失值和准确性的变化：

```python
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=10,
                    validation_data=validation_data.batch(512),
                    verbose=1)
```

```txt
Epoch 1/10
30/30 [==============================] - 4s 64ms/step - loss: 0.6693 - accuracy: 0.5195 - val_loss: 0.6309 - val_accuracy: 0.5810
Epoch 2/10
30/30 [==============================] - 2s 53ms/step - loss: 0.5751 - accuracy: 0.6606 - val_loss: 0.5362 - val_accuracy: 0.7132
Epoch 3/10
30/30 [==============================] - 2s 53ms/step - loss: 0.4485 - accuracy: 0.7926 - val_loss: 0.4263 - val_accuracy: 0.8026
Epoch 4/10
30/30 [==============================] - 2s 54ms/step - loss: 0.3241 - accuracy: 0.8724 - val_loss: 0.3589 - val_accuracy: 0.8408
Epoch 5/10
30/30 [==============================] - 2s 54ms/step - loss: 0.2387 - accuracy: 0.9146 - val_loss: 0.3256 - val_accuracy: 0.8549
Epoch 6/10
30/30 [==============================] - 2s 52ms/step - loss: 0.1791 - accuracy: 0.9420 - val_loss: 0.3113 - val_accuracy: 0.8654
Epoch 7/10
30/30 [==============================] - 2s 54ms/step - loss: 0.1312 - accuracy: 0.9613 - val_loss: 0.3052 - val_accuracy: 0.8676
Epoch 8/10
30/30 [==============================] - 2s 52ms/step - loss: 0.0965 - accuracy: 0.9741 - val_loss: 0.3084 - val_accuracy: 0.8666
Epoch 9/10
30/30 [==============================] - 2s 52ms/step - loss: 0.0700 - accuracy: 0.9851 - val_loss: 0.3166 - val_accuracy: 0.8693
Epoch 10/10
30/30 [==============================] - 2s 52ms/step - loss: 0.0500 - accuracy: 0.9913 - val_loss: 0.3297 - val_accuracy: 0.8665
```

## 6. 评估模型

评估模型：

```python
results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))
```

```txt
49/49 - 2s - loss: 0.3570 - accuracy: 0.8517 - 2s/epoch - 31ms/step
loss: 0.357
accuracy: 0.852
```

这个相当简单的模型准确率约 85%，使用更先进的方法，模型的准确率可达 95%。

## 7. 参考

- https://www.tensorflow.org/tutorials/keras/text_classification_with_hub
