# 图像分类基础：服装图像分类

- [图像分类基础：服装图像分类](#图像分类基础服装图像分类)
  - [1. 简介](#1-简介)
  - [2. 导入 Fashion MNIST 数据集](#2-导入-fashion-mnist-数据集)
  - [3. 查看数据](#3-查看数据)
  - [4. 数据预处理](#4-数据预处理)
  - [5. 构建模型](#5-构建模型)
    - [5.1 配置网络层](#51-配置网络层)
    - [5.2 编译模型](#52-编译模型)
  - [6. 训练模型](#6-训练模型)
    - [6.1 导入数据](#61-导入数据)
    - [6.2 评估准确性](#62-评估准确性)
    - [6.3 预测](#63-预测)
    - [6.4 验证预测结果](#64-验证预测结果)
  - [7. 使用模型](#7-使用模型)
  - [8. 参考](#8-参考)

Last updated: 2022-06-16, 17:04
@author Jiawei Mao
****

## 1. 简介

下面介绍训练神经网络模型对运动鞋（sneaker）和衬衫（shirt）等服装图像进行分类。如果有些细节无法理解也没关系，主要是为了建立对 TensorFlow 完整流程的感性认识。

下面使用 `tf.keras` 高级 API 构建模型。

```python
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
```

```txt
2.9.1
```

## 2. 导入 Fashion MNIST 数据集

Fashion MNIST 数据集包含 10 个类别共 70,000 张灰度图片，均为低分辨率（28*28 pixels）的服装图片，如下图所示：

![](2021-12-30-16-03-49.png)

Fashion MNIST 旨在替代经典的 MNIST 数据集。MNIST 经常被用作计算机视觉机器学习程序的 "Hello, World"。该数据集包含手写数字（0, 1, 2,...,9）图像，格式与 Fashion MNIST 一样。

对 Fashion MNIST 数据集进行分类比 MNIST 更难一点。这两个数据集都相对较小，适合用来验证算法，是测试和调试代码很好的起点。

使用 60,000 张图像训练模型，10,000 张图片评估模型。TensorFlow 内嵌有 Fashion MNIST 数据集，可以直接加载：

```python
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```

```txt
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
29515/29515 [==============================] - 0s 4us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
26421880/26421880 [==============================] - 1s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
5148/5148 [==============================] - 0s 0s/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
4422102/4422102 [==============================] - 1s 0us/step
```

加载数据集返回 4 个 NumPy 数组：

- `train_images` 和 `train_labels` 数组为训练集
- `test_images` 和 `test_labels` 数组为测试集

图像均为 28x28 NumPy 数组，像素值范围 [0,255]。标签为整数数组，值从 0 到 9。标签和类别的对应关系如下：

|标签|类别|
|----|---|
|0   |T-shirt/top|
|1   |Trouser    |
|2   |Pullover   |
|3   |Dress      |
|4   |Coat|
|5   |Sandal|
|6   |Shirt|
|7   |Sneaker|
|8   |Bag|
|9   |Ankle boot|

每个图像映射到一个类别。由于数据集中不包含类名，所以先将类别名称存储起来，方便后面绘图：

```python
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```

## 3. 查看数据

在训练模型前先熟悉一下数据集的格式。训练集包含 60,000 张图像，每张图像为 28x28 像素：

```python
train_images.shape
```

```txt
(60000, 28, 28)
```

同样，训练集包含 60,000 个标签：

```python
len(train_labels)
```

```txt
60000
```

每个标签都是 0 到 9 之间的整数：

```python
train_labels
```

```txt
array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)
```

测试集包含 10,000 张图像，每个图像同样是 28x28 像素：

```python
test_images.shape
```

```txt
(10000, 28, 28)
```

测试集包含 10,000 个标签：

```python
len(test_labels)
```

```txt
10000
```

## 4. 数据预处理

在训练模型前，要对数据进行预处理。图像的像素值在 0 到 255 之间，下面显示第一张训练图像：

```python
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show())
```

![](2021-12-30-18-55-39.png)

将这些像素值输入到神经网络模型之前，需要缩放到 0 到 1 之间。为此，将像素值除以 255。训练集和测试集以同样的方式处理：

```python
train_images = train_images / 255.0
test_images = test_images / 255.0
```

为了验证数据格式是否正确，下面显示训练集的前 25 个图像，并显示相应类别：

```python
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
```

![](2021-12-30-18-58-04.png)

## 5. 构建模型

构建神经网络模型：配置好模型的每一层，然后编译模型。

### 5.1 配置网络层

神经网络的基本单元是 layer。layer 从输入数据中提取表示（representation），并期望这些表示对当前问题是有意义的。

大多数神经网络都是将简单的 layer 连在一起。而大多数 layer，如 `tf.keras.layers.Dense` 包含需要在训练中学习的参数。

```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
```

网络的第一层 `tf.keras.layers.Flatten` 将图像从二维数组（$28\times 28$）转换为一维数组（$28\times 28=784$）。可以将 `Flatten` 的功能想象为将图像的像素一行行的拿下来连成一行。该 layer 没有需要学习的参数，单纯用于格式化数据。

格式化数据后，下面是两个 `tf.keras.layers.Dense` 层，即全连接层。第一个 `Dense` 层包含 128 个节点（或神经元），第二个 `Dense` 层返回长度为 10 的 logits 数组，每个节点包含一个与图像所属类别相关的打分值。

> logit 指神经网络的原始预测值，用 argmax 函数处理可以获得预测的类别，用 softmax 处理则获得概率值。

### 5.2 编译模型

在训练模型前，还需要进行一些设置。在编译步骤添加如下配置：

- **损失函数**（loss function），用于衡量训练过程中模型的准确性。通过最小化该函数的值引导模型往正确的方向演化；
- **优化器**（optimizer），模型根据看到的数据和损失函数进行更新的方式；
- **评价指标**（metrics），用来监控训练和测试步骤的评价指标。例如，下面使用 `accuracy`，即正确分类图片的比例作为评价指标。

```python
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

## 6. 训练模型

按如下步骤训练神经网络模型：

1. 将训练数据导入模型。在本例中，训练数据在 `train_images` 和 `train_labels` 数组中；
2. 模型学习如何将图像和标签关联起来；
3. 让模型预测测试集，在本例中为 `test_images` 数组；
4. 验证预测结果是否和标签 `test_labels` 匹配。

### 6.1 导入数据

使用 `model.fit` 方法开始训练：

```python
model.fit(train_images, train_labels, epochs=10)
```

```txt
Epoch 1/10
1875/1875 [==============================] - 6s 3ms/step - loss: 0.4928 - accuracy: 0.8258
Epoch 2/10
1875/1875 [==============================] - 5s 3ms/step - loss: 0.3692 - accuracy: 0.8670
Epoch 3/10
1875/1875 [==============================] - 5s 2ms/step - loss: 0.3298 - accuracy: 0.8798
Epoch 4/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.3104 - accuracy: 0.8865
Epoch 5/10
1875/1875 [==============================] - 5s 2ms/step - loss: 0.2923 - accuracy: 0.8921
Epoch 6/10
1875/1875 [==============================] - 5s 2ms/step - loss: 0.2773 - accuracy: 0.8979
Epoch 7/10
1875/1875 [==============================] - 5s 2ms/step - loss: 0.2664 - accuracy: 0.9006
Epoch 8/10
1875/1875 [==============================] - 5s 2ms/step - loss: 0.2545 - accuracy: 0.9050
Epoch 9/10
1875/1875 [==============================] - 5s 2ms/step - loss: 0.2459 - accuracy: 0.9086
Epoch 10/10
1875/1875 [==============================] - 5s 2ms/step - loss: 0.2371 - accuracy: 0.9103
<keras.callbacks.History at 0x24db97cd490>
```

在训练过程中会输出损失值和评价指标。该模型对训练集的准确率约为 0.91，即 91 %。

### 6.2 评估准确性

接下来，测试模型在测试集上的表现：

```python
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
```

```sh
313/313 - 1s - loss: 0.3287 - accuracy: 0.8862 - 850ms/epoch - 3ms/step

Test accuracy: 0.8862000107765198
```

结果表面，模型在测试集上的精度略低于在训练集上的精度。这种训练集和测试集之间的精度差异表示**过拟合**（overfitting）。当机器学习模型在新的、之前未见过的数据上表现不如在训练集上的好，就是发生了过拟合。过拟合模型“记住”了训练集上的噪音和细节，从而对模型在新数据上的性能产生负面影响。

### 6.3 预测

模型训练好后，可以用它预测新的图像。在模型的线性输出 logits 后添加 `softmax` 层，以将 logits 转换为易于理解的概率值：

```python
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
```

```python
predictions = probability_model.predict(test_images)
```

模型预测了测试集中每个图像的标签。先看看第一个预测值：

```python
predictions[0]
```

```sh
array([1.3177932e-06, 5.2332280e-07, 1.4412682e-07, 3.9305362e-10,
       5.2675224e-09, 1.7111773e-02, 3.0795927e-06, 1.9957772e-03,
       2.0979488e-07, 9.8088717e-01], dtype=float32)
```

预测结果为长度为 10 的数组，每个数字代表了图像属于该类别的概率。查看概率值最大的标签：

```python
np.argmax(predictions[0])
```

```txt
9
```

因此，模型认为这个图像是一个 ankle boot，即 `class_names[9]`。检查其标签发现该分类正确：

```python
test_labels[0]
```

```txt
9
```

绘图查看 10 类分别预测的准确率：

```python
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
```

### 6.4 验证预测结果

模型训练后，可以用它对其它图像进行预测。

先看对第一张图像的预测。正确的预测标签用蓝色表示，错误的用红色表示。数值给出正确预测的概率百分比。

```python
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()
```

![](2022-06-16-16-49-01.png)

```python
i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()
```

![](2022-06-16-16-50-25.png)

下面绘制多张图片及其预测值。可以发现，即使预测的概率值很高，预测结果也可能是错的。

```python
# 绘制前 15 张测试图片、预测标签和真实标签
# 预测正确用蓝色表示，预测错误用红色表示
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
```

![](2022-06-16-16-57-56.png)

## 7. 使用模型

最后，使用模型预测单个图片。从测试集抽出一张图片：

```python
img = test_images[1]

print(img.shape)
```

```txt
(28, 28)
```

`tf.keras` 模型对批量样本的预测进行的优化。因此，即使只预测一张图像，也要将它添加到 list 中：

```python
img = (np.expand_dims(img, 0))

print(img.shape)
```

```txt
(1, 28, 28)
```

然后预测图像标签：

```python
predictions_single = probability_model.predict(img)

print(predictions_single)
```

```txt
1/1 [==============================] - 0s 20ms/step
[[7.3736344e-05 6.2409075e-16 9.9798298e-01 1.3548121e-10 1.1947579e-03
  1.4547930e-11 7.4847863e-04 9.6510517e-13 1.6704501e-11 9.4280605e-13]]
```

```python
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()
```

![](2022-01-01-15-06-55.png)

`tf.keras.Model.predict` 返回 list of list，每个图像对应一个 list。查看我们唯一一张图片的预测值：

```python
np.argmax(predictions_single[0])
```

```txt
2
```

和预期结果一致。

## 8. 参考

- https://www.tensorflow.org/tutorials/keras/classification
