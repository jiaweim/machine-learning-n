# 数据增强

- [数据增强](#数据增强)
  - [简介](#简介)
  - [配置](#配置)
  - [下载数据集](#下载数据集)
  - [使用 Keras 预处理层](#使用-keras-预处理层)
    - [调整大小和缩放](#调整大小和缩放)
    - [数据增强之 Keras](#数据增强之-keras)
    - [使用 Keras 预处理层的两种方式](#使用-keras-预处理层的两种方式)
  - [参考](#参考)

2022-01-14, 13:10
@author Jiawei Mao
***

## 简介

下面演示如何使用 TensorFlow 执行数据增强，即使用随机（现实存在的）变换（如旋转）来增加训练集多样性的技术。

在 TensorFlow 中实现数据增强的方式有两种：

- 使用 Keras 预处理层，如 `tf.keras.layers.Resizing`, `tf.keras.layers.Rescaling`, `tf.keras.layers.RandomFlip` 以及 `tf.keras.layers.RandomRotation`；
- 使用 `tf.image` 的方法，如 `tf.image.flip_left_right`, `tf.image.rgb_to_grayscale`, `tf.image.adjust_brightness`, `tf.image.central_crop` 以及 `tf.image.stateless_random*` 等方法。

## 配置

```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras import layers
```

## 下载数据集

下面使用 [tf_flowers](https://www.tensorflow.org/datasets/catalog/tf_flowers) 数据集演示。

```python
(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)
```

该花卉数据集包含 5 个类别：

```python
num_classes = metadata.features['label'].num_classes
print(num_classes)
```

```sh
5
```

我们从数据集中取出一张图片，演示数据增强的效果。

```python
get_label_name = metadata.features['label'].int2str

image, label = next(iter(train_ds))
_ = plt.imshow(image)
_ = plt.title(get_label_name(label))
```

> 此处抛出了错误，添加以下代码可以避免该错误。

```python
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
```

![](2022-01-14-14-15-59.png)

## 使用 Keras 预处理层

### 调整大小和缩放

可以使用 Keras 预处理层调整图像大小（`tf.keras.layers.Resizing`）和像素值大小（`tf.keras.layers.Rescaling`）：

```python
IMG_SIZE = 180

resize_and_rescale = tf.keras.Sequential([
  layers.Resizing(IMG_SIZE, IMG_SIZE),
  layers.Rescaling(1./255)
])
```

> 上面的缩放层（`Rescaling`） 将像素值调整到 [0, 1] 之间。如果希望调整到 [-1, 1] 之间，可以使用 `tf.keras.layers.Rescaling(1./127.5, offset=-1)`。

可以查看调整后的效果：

```python
result = resize_and_rescale(image)
_ = plt.imshow(result)
```

![](2022-01-14-14-52-27.png)

查看像素值是否在 `[0, 1]` 之间：

```python
print("Min and max pixel values:", result.numpy().min(), result.numpy().max())
```

```sh
Min and max pixel values: 0.0 1.0
```

### 数据增强之 Keras

可以使用 Keras 预处理层进行数据增强。例如，`tf.keras.layers.RandomFlip` 和 `tf.keras.layers.RandomRotation`，

下面创建一些预处理层，并将它们重复应用到同一图像。

```python
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])
```

将图像添加到 batch：

```python
# Add the image to a batch.
image = tf.expand_dims(image, 0)
```

```python
plt.figure(figsize=(10, 10))
for i in range(9):
  augmented_image = data_augmentation(image)
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(augmented_image[0])
  plt.axis("off")
```

![](output.png)

还有很多用于数据增强的预处理层，包括 `tf.keras.layers.RandomContrast`, `tf.keras.layers.RandomCrop` 以及 `tf.keras.layers.RandomZoom` 等。

### 使用 Keras 预处理层的两种方式

使用 Keras 预处理层的方式有两种。

- 方法一：预处理层作为模型的一部分

```python
model = tf.keras.Sequential([
  # Add the preprocessing layers you created earlier.
  resize_and_rescale,
  data_augmentation,
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  # Rest of your model.
])
```

使用该方式有两个需要注意：

1. 数据增强和其它的网络层在相同的设备运行，因此也会受益于 GPU 加速；
2. 当使用 `model.save` 导出模型时，预处理层也会一起保存。后面部署模型时，它会自动对图像进行预处理，这样就不用在服务器端重新实现预处理逻辑。

> 数据增强层在测试时不活动，输入图像只在调用 `Model.fit` 时被增强，在调用 `Model.evaluate` 或 `Model.predict` 时

## 参考

- https://www.tensorflow.org/tutorials/images/data_augmentation
