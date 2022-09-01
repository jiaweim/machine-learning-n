# Keras Tuner 简介

## 概述

Keras Tuner 是一个库，用于辅助位 TF 程序选择最佳的超参数。为机器学习模型选择合适超参数集的过程称为超参数调整（hyperperameter tuning or *hypertuning*）。

超参数是控制 ML 模型的拓扑结构以及训练过程的变量。这些变量在整个训练过程中保持不变，但直接影响 ML 程序的性能。超参数有两种类型：

1. **模型超参数**，影响模型选择，如隐藏层的数量和宽度；
2. **算法超参数**，影响学习算法的速度和质量，如随机梯度下降（SGD）的学习率，KNN 分类器的最邻近数量。

本教程演示使用 Keras Tuner 为图像分类程序执行 hypertuning。

## 配置

```python
import tensorflow as tf
from tensorflow import keras
```

安装并导入 Keras Tuner：

```powershell
pip install -q -U keras-tuner
```

```python
import keras_tuner as kt
```

## 准备数据

下面使用 [Fashion MNIST 数据集](https://github.com/zalandoresearch/fashion-mnist) 训练一个对服装图像进行分类的模型，使用 Keras Tuner 找到最佳的超参数。

载入数据

```python
(img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()
```

```python
# Normalize pixel values between 0 and 1
img_train = img_train.astype("float32") / 255.0
img_test = img_test.astype("float32") / 255.0
```

## 定义模型

在构建模型进行 hypertuning 时，除了定义模型架构外，还要定义超参数的搜索空间。为 hypertuning 设置的模型称为 hypermodel。

定义 hypermodel 的方法有两种：

- 使用模型构建函数
- 继承 Keras Tuner 的 `HyperModel` 类

还可以使用两个预定义的用于计算机视觉的 [HyperModel](https://keras.io/api/keras_tuner/hypermodels/) 类：[HyperXception](https://keras.io/api/keras_tuner/hypermodels/hyper_xception/) 和 [HyperResNet](https://keras.io/api/keras_tuner/hypermodels/hyper_resnet/)。

本教程使用模型构建函数定义图像分类模型。模型构建函数返回一个编译的模型，并使用呢欸连定义的 hyperparameters 来 hypertune 模型。

```python

```

## 参考

- https://www.tensorflow.org/tutorials/keras/keras_tuner
