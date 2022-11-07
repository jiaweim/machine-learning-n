# 函数 API

- [函数 API](#函数-api)
  - [配置](#配置)
  - [简介](#简介)
  - [训练、评估和推断](#训练评估和推断)
  - [保存和序列化](#保存和序列化)
  - [使用相同 layers 定义多个模型](#使用相同-layers-定义多个模型)
  - [模型是可调用对象](#模型是可调用对象)
  - [复杂拓扑图形结构](#复杂拓扑图形结构)
    - [多输入输出](#多输入输出)
    - [7.2 A toy ResNet model](#72-a-toy-resnet-model)
  - [8. 共享层](#8-共享层)
  - [9. 提取和重用 graph 节点](#9-提取和重用-graph-节点)
  - [10. 自定义 layer](#10-自定义-layer)
  - [11. 何时使用函数 API](#11-何时使用函数-api)
    - [11.1 函数 API 优势](#111-函数-api-优势)
    - [11.2 函数 API 缺点](#112-函数-api-缺点)
  - [12. API 混搭](#12-api-混搭)
  - [13. 参考](#13-参考)

Last updated: 2022-06-30, 15:36
@author Jiawei Mao
****

## 配置

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

## 简介

Keras 函数（functional）API 是一种比 [tf.keras.Sequential](https://tensorflow.google.cn/guide/keras/sequential_model) 更灵活的创建模型的方法。函数式 API 可以创建非线性拓扑结构、共享层，以及包含多个输入或多个输出的模型。

其主要思想是，深度学习模型是由神经网络 layer 组成的有向无环图（directed acyclic graph, DAG），函数式 API 提供构建这种图的方法。

考虑如下模型：

```python
(input: 784-dimensional vectors)
       ↧
[Dense (64 units, relu activation)]
       ↧
[Dense (64 units, relu activation)]
       ↧
[Dense (10 units, softmax activation)]
       ↧
(output: logits of a probability distribution over 10 classes)
```

该模型包含三层，使用函数 API 创建该模型，首先创建输入节点：

```python
inputs = keras.Input(shape=(784,))
```

输入向量 shape 为 784。此处一般只指定样本 shape，忽略 batch size。假如输入 shape 为 `(32, 32, 3)` 的图片。此时可定义输入为：

```python
img_inputs = keras.Input(shape=(32, 32, 3))
```

`keras.Input` 返回的 `inputs` 包含输入数据的 `shape` 和 `dtype`：

```python
>>> inputs.shape
TensorShape([None, 784])
>>> inputs.dtype
tf.float32
```

定义 DAG 的下一个节点，`inputs` 对象作为输入：

```python
dense = layers.Dense(64, activation="relu")
x = dense(inputs)
```

层调用（layer call）就像从 "inputs" 到这个 dense 层画了个箭头。将 `inputs` 传入 `dense` 层，获得输出 `x`。

继续创建 graph 的余下两层：

```python
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(10)(x)
```

此时可以将 `inputs` 和 `outputs` 作为参数创建 `Model`：

```python
model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
```

- 查看模型

```python
model.summary()
```

```txt
Model: "mnist_model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 784)]             0         
                                                                 
 dense (Dense)               (None, 64)                50240     
                                                                 
 dense_1 (Dense)             (None, 64)                4160      
                                                                 
 dense_2 (Dense)             (None, 10)                650       
                                                                 
=================================================================
Total params: 55,050
Trainable params: 55,050
Non-trainable params: 0
_________________________________________________________________
```

- 将模型结构输出为图片

```python
keras.utils.plot_model(model, "my_first_model.png")
```

> 导出图片需要安装 [Graphviz](https://graphviz.org/)，同时安装 python 包 pydot

![](images/2022-02-16-12-10-57.png)

- 导出图片时可以显示每个 layer 的输入和输出 shape

```python
keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)
```

![](images/2022-02-16-12-31-53.png)

这个图和创建该图的代码几乎相同，只是将连接箭头用方法调用替换了。

## 训练、评估和推断

使用函数 API 构建的模型，在训练、评估和推断方面与 [Sequential](https://tensorflow.google.cn/guide/keras/sequential_model) 模型完全相同。

`Model` 类内置训练循环方法 `fit()` 和评估循环方法 `evaluate()`，并且可以自定义这些循环，以实现监督学习以外的训练流程，如 GAN。

下面，加载 MNIST 数据集，reshape 为向量，训练上面创建的模型，在 validation split 上监视性能，并使用测试集评估模型：

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)

test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
```

```txt
Epoch 1/2
750/750 [==============================] - 1s 1ms/step - loss: 0.3340 - accuracy: 0.9047 - val_loss: 0.1792 - val_accuracy: 0.9486
Epoch 2/2
750/750 [==============================] - 1s 907us/step - loss: 0.1558 - accuracy: 0.9530 - val_loss: 0.1485 - val_accuracy: 0.9569
313/313 - 0s - loss: 0.1512 - accuracy: 0.9560 - 152ms/epoch - 486us/step
Test loss: 0.15118415653705597
Test accuracy: 0.9559999704360962
```

## 保存和序列化

使用函数式 API 创建的模型的保存与序列化方式与使用 [Sequential](sequential_model.md) 创建的模型相同。保存函数式 API 创建的模型的标准方法是调用 `model.save()` 将整个模型保存为单个文件。随后可以用该文件重新创建完全相同的模型。

该文件包括：

- 模型结构
- 模型权重值（训练时学到的参数）
- 模型训练配置（传递给 `compile` 的参数）
- optimizer 及其状态（方便从训练中断的位置重新开始训练）

保存模型：

```python
model_path = r"D:\it\test\model"
model.save(model_path)
```

```txt
INFO:tensorflow:Assets written to: D:\it\test\model\assets
```

然后重新载入模型：

```python
del model
model = keras.models.load_model(model_path)
```

## 使用相同 layers 定义多个模型

在函数 API 中，通过指定 graph 的输入和输出创建模型。这意味着单个 graph of layers 可用来生成多个模型。

下面演示使用相同的 layers 堆栈实例化两个模型：

- 一个将输入图像转换为 16 维向量 `encoder` 模型
- 一个用于训练的端到端 autoencoder。

```python
encoder_input = keras.Input(shape=(28, 28, 1), name='img')
x = layers.Conv2D(16, 3, activation='relu')(encoder_input)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.Conv2D(16, 3, activation='relu')(x)
encoder_output = layers.GlobalMaxPooling2D()(x)

encoder = keras.Model(encoder_input, encoder_output, name='encoder')
encoder.summary()
```

```txt
Model: "encoder"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 img (InputLayer)            [(None, 28, 28, 1)]       0         
                                                                 
 conv2d (Conv2D)             (None, 26, 26, 16)        160       
                                                                 
 conv2d_1 (Conv2D)           (None, 24, 24, 32)        4640      
                                                                 
 max_pooling2d (MaxPooling2D  (None, 8, 8, 32)         0         
 )                                                               
                                                                 
 conv2d_2 (Conv2D)           (None, 6, 6, 32)          9248      
                                                                 
 conv2d_3 (Conv2D)           (None, 4, 4, 16)          4624      
                                                                 
 global_max_pooling2d (Globa  (None, 16)               0         
 lMaxPooling2D)                                                  
                                                                 
=================================================================
Total params: 18,672
Trainable params: 18,672
Non-trainable params: 0
_________________________________________________________________
```

```python
x = layers.Reshape((4, 4, 1))(encoder_output)
x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
x = layers.Conv2DTranspose(32, 3, activation='relu')(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation="relu")(x)

autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')
autoencoder.summary()
```

```sh
Model: "autoencoder"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 img (InputLayer)            [(None, 28, 28, 1)]       0         
                                                                 
 conv2d (Conv2D)             (None, 26, 26, 16)        160       
                                                                 
 conv2d_1 (Conv2D)           (None, 24, 24, 32)        4640      
                                                                 
 max_pooling2d (MaxPooling2D  (None, 8, 8, 32)         0         
 )                                                               
                                                                 
 conv2d_2 (Conv2D)           (None, 6, 6, 32)          9248      
                                                                 
 conv2d_3 (Conv2D)           (None, 4, 4, 16)          4624      
                                                                 
 global_max_pooling2d (Globa  (None, 16)               0         
 lMaxPooling2D)                                                  
                                                                 
 reshape (Reshape)           (None, 4, 4, 1)           0         
                                                                 
 conv2d_transpose (Conv2DTra  (None, 6, 6, 16)         160       
 nspose)                                                         
                                                                 
 conv2d_transpose_1 (Conv2DT  (None, 8, 8, 32)         4640      
 ranspose)                                                       
                                                                 
 up_sampling2d (UpSampling2D  (None, 24, 24, 32)       0         
 )                                                               
                                                                 
 conv2d_transpose_2 (Conv2DT  (None, 26, 26, 16)       4624      
 ranspose)                                                       
                                                                 
 conv2d_transpose_3 (Conv2DT  (None, 28, 28, 1)        145       
 ranspose)                                                       
                                                                 
=================================================================
Total params: 28,241
Trainable params: 28,241
Non-trainable params: 0
_________________________________________________________________
```

这里解码架构与编码架构严格对称，因此输出形状和输入形状 `(28, 28, 1)` 相同。

`Conv2D` layer 的逆操作是 `Conv2DTranspose` layer，`MaxPooling2D` 的逆操作是 `UpSampling2D` 层。

## 模型是可调用对象

模型和 layer 一样可调用，通过调用模型，不仅可以重用模型结构，还可以重用它的权重。

为了演示模型调用的效果，下面用另一个 autoencoder 示例演示：创建一个 encoder 模型和一个 decoder 模型，然后通过两次调用将它们连接起来获得最终的 autoencoder 模型：

```python
encoder_input = keras.Input(shape=(28, 28, 1), name="original_img")
x = layers.Conv2D(16, 3, activation="relu")(encoder_input)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.Conv2D(16, 3, activation="relu")(x)
encoder_output = layers.GlobalMaxPooling2D()(x)

encoder = keras.Model(encoder_input, encoder_output, name="encoder")
encoder.summary()
```

```txt
Model: "encoder"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
original_img (InputLayer)    [(None, 28, 28, 1)]       0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 26, 26, 16)        160       
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 24, 24, 32)        4640      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 8, 8, 32)          0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 6, 6, 32)          9248      
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 4, 4, 16)          4624      
_________________________________________________________________
global_max_pooling2d_1 (Glob (None, 16)                0         
=================================================================
Total params: 18,672
Trainable params: 18,672
Non-trainable params: 0
_________________________________________________________________
```

```python
decoder_input = keras.Input(shape=(16,), name="encoded_img")
x = layers.Reshape((4, 4, 1))(decoder_input)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu")(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation="relu")(x)

decoder = keras.Model(decoder_input, decoder_output, name="decoder")
decoder.summary()
```

```txt
Model: "decoder"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
encoded_img (InputLayer)     [(None, 16)]              0         
_________________________________________________________________
reshape_1 (Reshape)          (None, 4, 4, 1)           0         
_________________________________________________________________
conv2d_transpose_4 (Conv2DTr (None, 6, 6, 16)          160       
_________________________________________________________________
conv2d_transpose_5 (Conv2DTr (None, 8, 8, 32)          4640      
_________________________________________________________________
up_sampling2d_1 (UpSampling2 (None, 24, 24, 32)        0         
_________________________________________________________________
conv2d_transpose_6 (Conv2DTr (None, 26, 26, 16)        4624      
_________________________________________________________________
conv2d_transpose_7 (Conv2DTr (None, 28, 28, 1)         145       
=================================================================
Total params: 9,569
Trainable params: 9,569
Non-trainable params: 0
_________________________________________________________________
```

```python
autoencoder_input = keras.Input(shape=(28, 28, 1), name="img")
encoded_img = encoder(autoencoder_input)
decoded_img = decoder(encoded_img)
autoencoder = keras.Model(autoencoder_input, decoded_img, name="autoencoder")
autoencoder.summary()
```

```txt
Model: "autoencoder"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
img (InputLayer)             [(None, 28, 28, 1)]       0         
_________________________________________________________________
encoder (Functional)         (None, 16)                18672     
_________________________________________________________________
decoder (Functional)         (None, 28, 28, 1)         9569      
=================================================================
Total params: 28,241
Trainable params: 28,241
Non-trainable params: 0
_________________________________________________________________
```

如上所示，模型可以嵌套，即一个模型可以包含子模型。模型嵌套常用于模型集成。例如，将一组模型组合成一个单一模型，来平均它们的预测：

```python
def get_model():
    inputs = keras.Input(shape=(128,))
    outputs = layers.Dense(1)(inputs)
    return keras.Model(inputs, outputs)


model1 = get_model()
model2 = get_model()
model3 = get_model()

inputs = keras.Input(shape=(128,))
y1 = model1(inputs)
y2 = model2(inputs)
y3 = model3(inputs)
outputs = layers.average([y1, y2, y3])
ensemble_model = keras.Model(inputs=inputs, outputs=outputs)
```

## 复杂拓扑图形结构

### 多输入输出

函数 API 可以创建包含多个输入和输出的模型，[Sequential](sequential_model.md) API 就不行。

例如，如果你要构建一个系统，用于按优先级对客户的票据进行排序，并将其转到正确的部门，那么该模型至少有三个输入：

- 票据标题（文本输入）
- 票据内容（文本输入）
- 用户添加的标签（分类输入）

该模型包含两个输出：

- 优先级打分（0 到 1 之间，sigmoid 输出）
- 处理票据的部门（softmax 输出）

下面使用函数 API 构建该模型：

```python
num_tags = 12  # 标签个数
num_words = 10000  # 处理文本数据时的词汇量大小
num_departments = 4  # 部门数

title_input = keras.Input(
    shape=(None,), name="title"
)  # 变长标题序列
body_input = keras.Input(shape=(None,), name="body")  # 变长内容序列
tags_input = keras.Input(
    shape=(num_tags,), name="tags"
)  # 长度为 `num_tags` 的二进制向量

# 将标题嵌入到 64 维向量
title_features = layers.Embedding(num_words, 64)(title_input)
# 将内容嵌入到 64 维向量
body_features = layers.Embedding(num_words, 64)(body_input)

# 将标题中嵌入的单词序列简化为 128 维向量
title_features = layers.LSTM(128)(title_features)
# 将内容中嵌入的单词序列简化为 32 维向量
body_features = layers.LSTM(32)(body_features)

# 通过串联将所有特征合并到一个向量
x = layers.concatenate([title_features, body_features, tags_input])

# 预测优先级的回归层
priority_pred = layers.Dense(1, name="priority")(x)
# 部门分类功能
department_pred = layers.Dense(num_departments, name="department")(x)

# 实例化一个预测优先级和部门的端到端模型
model = keras.Model(
    inputs=[title_input, body_input, tags_input],
    outputs=[priority_pred, department_pred],
)
```

模型的图示：

```python
keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
```

![](images/2022-02-24-16-05-54.png)

在编译这类模型时，不同输出可以采用不同的损失函数。甚至可以为不同损失分配不同的权重，以调整它们对总训练损失的贡献：

```python
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[
        keras.losses.BinaryCrossentropy(from_logits=True),
        keras.losses.CategoricalCrossentropy(from_logits=True),
    ],
    loss_weights=[1.0, 0.2],
)
```

因为两个输出层的名称不同，所以也可以根据名称指定损失函数和权重：

```python
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={
        "priority": keras.losses.BinaryCrossentropy(from_logits=True),
        "department": keras.losses.CategoricalCrossentropy(from_logits=True),
    },
    loss_weights={"priority": 1.0, "department": 0.2},
)
```

输入数据和目标值，通过 NumPy 数组传入：

```python
# 虚拟输入数据
title_data = np.random.randint(num_words, size=(1280, 10))
body_data = np.random.randint(num_words, size=(1280, 100))
tags_data = np.random.randint(2, size=(1280, num_tags)).astype("float32")

# 虚拟目标值
priority_targets = np.random.random(size=(1280, 1))
dept_targets = np.random.randint(2, size=(1280, num_departments))

model.fit(
    {"title": title_data, "body": body_data, "tags": tags_data},
    {"priority": priority_targets, "department": dept_targets},
    epochs=2,
    batch_size=32,
)
```

```txt
Epoch 1/2
40/40 [==============================] - 4s 17ms/step - loss: 1.3428 - priority_loss: 0.7025 - department_loss: 3.2012
Epoch 2/2
40/40 [==============================] - 1s 13ms/step - loss: 1.3294 - priority_loss: 0.6992 - department_loss: 3.1509
<keras.callbacks.History at 0x2438808d5b0>
```

当以 `Dataset` 对象调用 `fit`，要么生成 `([title_data, body_data, tags_data], [priority_targets, dept_targets])` tuple of list，或 `({'title': title_data, 'body': body_data, 'tags': tags_data}, {'priority': priority_targets, 'department': dept_targets})` a tuple of dict。

### 7.2 A toy ResNet model

除了多输入、多输出模型外，函数 API 还可以创建非线性拓扑结构模型，而 `Sequential` API 不行。

比如可以用来创建残差连接。下面用 CIFAR10 构建一个 toy ResNet 模型来演示：

```python
inputs = keras.Input(shape=(32, 32, 3), name="img")
x = layers.Conv2D(32, 3, activation="relu")(inputs)
x = layers.Conv2D(64, 3, activation="relu")(x)
block_1_output = layers.MaxPooling2D(3)(x)

x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
block_2_output = layers.add([x, block_1_output])

x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
block_3_output = layers.add([x, block_2_output])

x = layers.Conv2D(64, 3, activation="relu")(block_3_output)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs, outputs, name="toy_resnet")
model.summary()
```

```txt
Model: "toy_resnet"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 img (InputLayer)               [(None, 32, 32, 3)]  0           []                               
                                                                                                  
 conv2d (Conv2D)                (None, 30, 30, 32)   896         ['img[0][0]']                    
                                                                                                  
 conv2d_1 (Conv2D)              (None, 28, 28, 64)   18496       ['conv2d[0][0]']                 
                                                                                                  
 max_pooling2d (MaxPooling2D)   (None, 9, 9, 64)     0           ['conv2d_1[0][0]']               
                                                                                                  
 conv2d_2 (Conv2D)              (None, 9, 9, 64)     36928       ['max_pooling2d[0][0]']          
                                                                                                  
 conv2d_3 (Conv2D)              (None, 9, 9, 64)     36928       ['conv2d_2[0][0]']               
                                                                                                  
 add (Add)                      (None, 9, 9, 64)     0           ['conv2d_3[0][0]',               
                                                                  'max_pooling2d[0][0]']          
                                                                                                  
 conv2d_4 (Conv2D)              (None, 9, 9, 64)     36928       ['add[0][0]']                    
                                                                                                  
 conv2d_5 (Conv2D)              (None, 9, 9, 64)     36928       ['conv2d_4[0][0]']               
                                                                                                  
 add_1 (Add)                    (None, 9, 9, 64)     0           ['conv2d_5[0][0]',               
                                                                  'add[0][0]']                    
                                                                                                  
 conv2d_6 (Conv2D)              (None, 7, 7, 64)     36928       ['add_1[0][0]']                  
                                                                                                  
 global_average_pooling2d (Glob  (None, 64)          0           ['conv2d_6[0][0]']               
 alAveragePooling2D)                                                                              
                                                                                                  
 dense_3 (Dense)                (None, 256)          16640       ['global_average_pooling2d[0][0]'
                                                                 ]                                
                                                                                                  
 dropout (Dropout)              (None, 256)          0           ['dense_3[0][0]']                
                                                                                                  
 dense_4 (Dense)                (None, 10)           2570        ['dropout[0][0]']                
                                                                                                  
==================================================================================================
Total params: 223,242
Trainable params: 223,242
Non-trainable params: 0
```

模型图：

```python
keras.utils.plot_model(model, "mini_resnet.png", show_shapes=True)
```

![](images/2022-06-27-15-11-56.png)

训练模型：

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["acc"],
)
# 这里只使用训练集的前 1000 个样本以节省时间
model.fit(x_train[:1000], y_train[:1000], batch_size=64, epochs=1, validation_split=0.2)
```

```txt
13/13 [==============================] - 4s 47ms/step - loss: 2.3047 - acc: 0.1338 - val_loss: 2.2925 - val_acc: 0.1150
<keras.callbacks.History at 0x2450f855430>
```

## 8. 共享层

函数 API 的另一个重要用途是创建包含共享层的模型。共享层是同一模型中多次重复使用的层，它们学习与 GRAPH 中多条路径相关的特征。

共享层通常用于编码来自相似空间的输入（如，两段具有相似词汇的不同文本），它们能够在不同的输入之间共享信息，并且能够在相对较少的数据上训练这种模型。如果在其中一个输入中看到一个指定单词，其它通过共享层的输入处理都将受益。

在函数 API 中要共享某一层，只需要多次调用该层。例如，在两个不同的文本输入中共享 `Embedding` 层：

```python
# 将 1000 个 unique 单词映射到 128 维向量
shared_embedding = layers.Embedding(1000, 128)

# 变长整数序列
text_input_a = keras.Input(shape=(None,), dtype="int32")

# 变长整数序列
text_input_b = keras.Input(shape=(None,), dtype="int32")

# 使用相同 layer 编码两个输入
encoded_input_a = shared_embedding(text_input_a)
encoded_input_b = shared_embedding(text_input_b)
```

## 9. 提取和重用 graph 节点

layer graph 是静态数据结构，可以访问、检查其内容。这意味着可以访问 GRAPH 中间层的激活值，并在其它地方使用，如提取特征。

例如，使用 ImageNet 预训练权重 VGG19 模型：

```python
vgg19 = tf.keras.applications.VGG19()
```

```txt
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels.h5
574710816/574710816 [==============================] - 118s 0us/step
```

通过查询 GRAPH 数据结构，获得模型中间激活值：

```python
features_list = [layer.output for layer in vgg19.layers]
```

使用这些特征创建一个新的特征提取模型，该模型返回中间层激活值：

```python
feat_extraction_model = keras.Model(inputs=vgg19.xs, outputs=features_list)

img = np.random.random((1, 224, 224, 3)).astype("float32")
extracted_features = feat_extraction_model(img)
```

这种方法在 [风格迁移学习](https://keras.io/examples/generative/neural_style_transfer/) 中会用到。

## 10. 自定义 layer

`tf.keras` 包含各种内置 layers，例如：

- 卷积层：`Conv1D`, `Conv2D`, `Conv3D`, `Conv2DTranspose`
- 池化层：`MaxPooling1D`, `MaxPooling2D`, `MaxPooling3D`, `AveragePooling1D`
- RNN  层：`GRU`, `LSTM`, `ConvLSTM2D`
- `BatchNormalization`, `Dropout`, `Embedding` 等。

如果这些都不满足需求，还可以自定义层。所有的 layer 都扩展 `Layer` 类并实现：

- `call` 方法，指定 layer 执行的计算
- `build` 方法，创建 layer 的权重（代码风格建议，权重的创建也可以放在 `__init__`）

自定义 layer 详情，参考[自定义 layer 和 model 指南](custom_layers_and_models.md)。

下面是 `tf.keras.layers.Dense` 的基本实现：

```python
class CustomDense(layers.Layer):
    def __init__(self, units=32):
        super(CustomDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


inputs = keras.Input((4,))
outputs = CustomDense(10)(inputs)

model = keras.Model(inputs, outputs)
```

为了在自定义 layer 中支持序列化，需要实现 `get_config` 方法，该方法返回 layer 实例的构造函数参数：

```python
class CustomDense(layers.Layer):
    def __init__(self, units=32):
        super(CustomDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        return {"units": self.units}


inputs = keras.Input((4,))
outputs = CustomDense(10)(inputs)

model = keras.Model(inputs, outputs)
config = model.get_config()

new_model = keras.Model.from_config(config, custom_objects={"CustomDense": CustomDense})
```

还可以实现 `from_config(cls, config)` 类方法，该方法用于在给定配置 dict 下重新创建 layer 实例。`from_config` 的默认实现：

```python
def from_config(cls, config):
  return cls(**config)
```

## 11. 何时使用函数 API

是使用 Keras 函数 API 创建新模型，还是扩展 `Model` 类？一般来说，函数 API 更高级、更简单、更安全，具有许多扩展 `Model` 类不支持的特征。

然而，当构建的模型不好表示为有向无环图时，就只能扩展 `Model` 类，扩展 `Model` 类提供了更大的灵活性。例如，使用函数 API 无法实现 Tree-RNN，只能通过扩展 `Model` 类实现。

### 11.1 函数 API 优势

以下特征也适用于 `Sequential` 模型（也是数据结构），但不适用于扩展子类模型（Python 字节码，而非数据结构）。

**更简洁**

没有 `super(MyClass, self).__init__(...)`, `def call(self, ...):` 等定义。

- Keras 函数版本

```python
inputs = keras.Input(shape=(32,))
x = layers.Dense(64, activation='relu')(inputs)
outputs = layers.Dense(10)(x)
mlp = keras.Model(inputs, outputs)
```

- 扩展 `Model` 版本

```python
class MLP(keras.Model):

  def __init__(self, **kwargs):
    super(MLP, self).__init__(**kwargs)
    self.dense_1 = layers.Dense(64, activation='relu')
    self.dense_2 = layers.Dense(10)

  def call(self, inputs):
    x = self.dense_1(inputs)
    return self.dense_2(x)

# 实例化模型
mlp = MLP()
# 创建模型状态所需
# 模型至少被调用一次，才具有状态
_ = mlp(tf.zeros((1, 32)))
```

**定义连接图时验证模型**

在函数 API 中，输入规范（shape 和 dtype）提前使用 `Input` 指定。每次调用 一个 layer 时，该 layer 都会检查传递给它的输入是否符合要求，如果没有，就会发出有用的错误信息。

这样可以保证使用函数 API 创建的模型能运行。所有调试（与收敛相关的调试除外）都是在模型构建期间进行的，而不是执行期。这类似于编译器中的类型检查。

**函数模型可绘制、可检查**

可以将模型绘制为图像，并且可以轻松访问其中间节点。例如，提取、重用中间层的激活：

```python
features_list = [layer.output for layer in vgg19.layers]
feat_extraction_model = keras.Model(inputs=vgg19.xs, outputs=features_list)
```

**函数模型可以序列化和额克隆**

由于函数模型是数据结构而不是代码段，所以可以安全地序列化，也可以保存为单个文件，这样在不访问原始代码的情况下就可以重新创建完全相同的模型。

要序列化子类模型，必须在模型中实现 `get_config()` 和 `from_config()` 模型。

### 11.2 函数 API 缺点

**不支持动态体系结构**

函数 API 将模型视为 layer 的 DAG 图，对大多数深度学习结构确实如此，但不是全部，例如递归网络和 Tree-RNN 就不是 DAG 图，无法使用函数 API 实现。

## 12. API 混搭

函数 API 和扩展 `Model` 子类模型两种方法并不是二选一。`tf.keras` API 的所有模型都可以相互交互，即 `Sequential` 模型、函数模型和扩展 `Model` 子类模型都可以相互交互。

可以将函数模型或 `Sequential` 模型作为子类模型的一部分：

```python
units = 32
timesteps = 10
input_dim = 5

# 定义函数模型
inputs = keras.Input((None, units))
x = layers.GlobalAveragePooling1D()(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

class CustomRNN(layers.Layer):
    def __init__(self):
        super(CustomRNN, self).__init__()
        self.units = units
        self.projection_1 = layers.Dense(units=units, activation="tanh")
        self.projection_2 = layers.Dense(units=units, activation="tanh")
        # Our previously-defined Functional model
        self.classifier = model

    def call(self, inputs):
        outputs = []
        state = tf.zeros(shape=(inputs.shape[0], self.units))
        for t in range(inputs.shape[1]):
            x = inputs[:, t, :]
            h = self.projection_1(x)
            y = h + self.projection_2(state)
            state = y
            outputs.append(y)
        features = tf.stack(outputs, axis=1)
        print(features.shape)
        return self.classifier(features)


rnn_model = CustomRNN()
_ = rnn_model(tf.zeros((1, timesteps, input_dim)))
```

```txt
(1, 10, 32)
```

可以在函数 API 中使用子类 layer 或模型，只需要实现遵循以下模式的 `call` 方法：

- `call(self, inputs, **kwargs)`，其中 `inputs` 为张量或张量的嵌套结构，如张量列表，`**kwargs` 为非张量参数。
- `call(self, inputs, training=None, **kwargs)` `training` 为 boolean 值，表示该 layer 是在训练模式还是推理模式。
- `call(self, inputs, mask=None, **kwargs)`，`mask` 为 boolean 屏蔽张量，RNN 中很有用
- `call(self, inputs, training=None, mask=None, **kwargs)`，同时包含 `mask` 和 `training` 参数。

另外，可以在自定义 layer 或 model 上实现 `get_config`，这样创建的函数模型就可以序列化和克隆。

下面是从头开始自定义的 RNN 示例：

```python
units = 32
timesteps = 10
input_dim = 5
batch_size = 16

class CustomRNN(layers.Layer):
    def __init__(self):
        super(CustomRNN, self).__init__()
        self.units = units
        self.projection_1 = layers.Dense(units=units, activation="tanh")
        self.projection_2 = layers.Dense(units=units, activation="tanh")
        self.classifier = layers.Dense(1)

    def call(self, inputs):
        outputs = []
        state = tf.zeros(shape=(inputs.shape[0], self.units))
        for t in range(inputs.shape[1]):
            x = inputs[:, t, :]
            h = self.projection_1(x)
            y = h + self.projection_2(state)
            state = y
            outputs.append(y)
        features = tf.stack(outputs, axis=1)
        return self.classifier(features)


# 使用 `batch_shape` 指定输入静态 batch 大小
# 因为 `CustomRNN` 内部计算需要静态 batch size (创建 `state` 零张量时)
inputs = keras.Input(batch_shape=(batch_size, timesteps, input_dim))
x = layers.Conv1D(32, 3)(inputs)
outputs = CustomRNN()(x)

model = keras.Model(inputs, outputs)

rnn_model = CustomRNN()
_ = rnn_model(tf.zeros((1, 10, 5)))
```

## 13. 参考

- https://www.tensorflow.org/guide/keras/functional
