# Keras 简介（工程师）

- [Keras 简介（工程师）](#keras-简介工程师)
  - [简介](#简介)
  - [数据加载和预处理](#数据加载和预处理)
    - [加载数据](#加载数据)
    - [数据预处理](#数据预处理)
    - [Keras 预处理层](#keras-预处理层)
  - [使用 Keras 函数 API 构建模型](#使用-keras-函数-api-构建模型)

***

## 简介

介绍使用 Keras 构建真实世界的机器学习解决方法的所有内容。

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
```

如果你是一名机器学习工程师，希望使用 Keras 在真实产品中添加深度学习功能？本指南对核心 Keras API 概念进行介绍：

- 准备数据（转换为 NumPy 数组或 `tf.data.Dataset`）
- 数据预处理，例如特征归一化、词汇索引
- 使用 Keras 函数 API 构建模型
- 使用内置的 `Keras.fit()` 方法训练模型
- 在测试集上评估模型，以及用它来推断新数据
- 自定义 `fit()`，例如构建 GAN
- 利用多个 GPU 加快训练
- 调整超参数优化模型

## 数据加载和预处理

神经网络不能直接处理原始数据，如文本文件、编码的 JPEG 图像文件或 CSV 文件等。它们处理向量化、标准化的数据表示：

- 文本文件需要读取为字符串张量，然后拆分为单词。最后，需要将单词索引并装好为整数张量。
- 图像需要读取并解码为整数张量，然后转换为浮点数并归一化给更小的值（通常在 0 和 1 之间）
- CSV 数据需要解析，将数字特征转换为浮点张量，分类特征进行索引并转换为这鞥书张量。然后，每个特征通常需要归一化为均值为 0，方差为 1.

### 加载数据

Keras 模型接受三种类型的输入：

- NumPy 数组，和 Scikit-Learn 等基于 Python 库一样，如果数据能够放入 内存，这是个很好的选项。
- TensorFlow `Dataset` 对象，适合于无法放入内存的数据集
- Python generators 生成成批数据，例如 `keras.utils.Sequence` 的子类。

在开始训练模型之前，要将数据转换为上述格式之一。如果你的数据集很大，并且需要在 GPU 上进行训练，则建议使用 `Dataset` 对象，因为其性能更佳：

- 当 GPU 忙时，可以在 CPU 上异步预处理数据，并将其缓冲到队列中
- 在 GPU 显存上预取数据，以便在 GPU 完成一批数据处理时，立刻就有一批数据可用，以充分利用 GPU。

Keras 提供了许多使用程序，辅助将磁盘上的原始数据转换为 `Dataset`:

- `tf.keras.preprocessing.image_dataset_from_directory` 将按目录分类保存的图像转换为带标签的图像张量数据集
- `tf.keras.preprocessing.text_dataset_from_directory` 将按目录分类保存的文本转换为带标签的文本张量数据集

此外，TensorFlow `tf.data` 包含其它类似的实用程序，如 `tf.data.experimental.make_csv_dataset` 从 CSV 文件加载结构化数据。

**示例：从图片文件生成标记数据集**

假设你在不同文件夹下按类别保存图像文件，如下：

```txt
main_directory/
...class_a/
......a_image_1.jpg
......a_image_2.jpg
...class_b/
......b_image_1.jpg
......b_image_2.jpg
```

然后可以按如下操作：

```python
# 创建数据集
dataset = keras.preprocessing.image_dataset_from_directory(
  'path/to/main_directory', batch_size=64, image_size=(200, 200))

# 为了演示，迭代数据集生成的批次
for data, labels in dataset:
   print(data.shape)  # (64, 200, 200, 3)
   print(data.dtype)  # float32
   print(labels.shape)  # (64,)
   print(labels.dtype)  # int32
```

样本的标签是其所在文件夹按字母数字排序的位置。当然，也可以显式设置，例如 `class_names=['class_a', 'class_b']`，此时标签 `0` 对应 `class_a`，`1` 对应 `class_b`。

**示例：从文本文件生成标记数据集**

对文本文件操作一样：

```python
dataset = keras.preprocessing.text_dataset_from_directory(
  'path/to/main_directory', batch_size=64)

for data, labels in dataset:
   print(data.shape)  # (64,)
   print(data.dtype)  # string
   print(labels.shape)  # (64,)
   print(labels.dtype)  # int32
```

### 数据预处理

将数据转换为 NumPy 数组或 `Dataset` 对象后，就可以对数据进行预处理了。即：

- 字符串数据的标记后，然后进行标记索引
- 特征归一化
- 将数据缩放到较小的值，神经网络中，一般输入值接近 0，通常缩放为均值为 0 ，方差为 1，或者直接缩放到 [0, 1] 之间。

通常，应该尽可能地将数据预处理作为模型的一部分，而不是通过外部数据通道处理。因为外部数据预处理使得模型在生产中使用的可移植性降低。例如，对一个文本预处理模型：它使用特定的标记化算法和特定的词汇表索引。当你想将模型发布到移动端或 JavaScript 程序时，就需要用目标语言重新创建相同的预处理过程。这可能非常棘手，原始管道和你重新创建的管道的微小差异都可能使你的模型完全无效，或者导致性能严重降低。

如果能够简单的导出包含预处理的端到端模型，就要容易得多。理想的模型应该接受输入尽可能接近原始数据的内容：对图像模型，应接受 [0, 255] 范围的 RGB 像素值，对文本模型，则应接受 UTF-8 字符串。这样，导出模型后，使用者就不必知道如何预处理了。

### Keras 预处理层

在 Keras 中，可以通过预处理层定义模型内数据预处理。包括：

- 使用 `TextVectorization` 对原始文本字符串进行向量化
- 通过 `Normalization` 对特征进行归一化
- 图像缩放、裁剪或数据增强

使用 Keras 预处理层的关键优势在于，它们可以在模型训练前或训练后直接包含在模型中，使得模型具有可移植性。

部分预处理层带有状态：

- `TextVectorization` 包含单词或标记到整数索引的映射
- `Normalization` 包含特征的均值和偏差

预处理层的状态（state）通过在训练集上调用 `layer.adapt(data)` 获得。

**示例：将字符串转换为整数索引**

```python
from tensorflow.keras.layers import TextVectorization

# 训练数据示例: dtype `string`.
training_data = np.array([["This is the 1st sample."], ["And here's the 2nd sample."]])

# 创建 TextVectorization layer 实例，可以返回整数标记索引或
# 密集标记表示
vectorizer = TextVectorization(output_mode="int")

# 调用 `adapt` 使 layer 生成与数据对应的词汇表索引
vectorizer.adapt(training_data)

# 调用 adapt 后，layer 能对之前在在 `adapt()` 数据中看到的任何
# n-gram 进行编码，未知的 n--gram 使用 "out-of-vocabulary" 表示
integer_data = vectorizer(training_data)
print(integer_data)
```

```bash
tf.Tensor(
[[4 5 2 9 3]
 [7 6 2 8 3]], shape=(2, 5), dtype=int64)
```

**示例：特征归一化**

```python
from tensorflow.keras.layers import Normalization

# Example image data, with values in the [0, 255] range
training_data = np.random.randint(0, 256, size=(64, 200, 200, 3)).astype("float32")

normalizer = Normalization(axis=-1)
normalizer.adapt(training_data)

normalized_data = normalizer(training_data)
print("var: %.4f" % np.var(normalized_data))
print("mean: %.4f" % np.mean(normalized_data))
```

```txt
var: 1.0000
mean: 0.0000
```

**示例：图像的缩放和中心裁剪**

`Rescaling` 和 `CenterCrop` layer 都是无状态的，因此无需调用 `adapt()`。

```python
from tensorflow.keras.layers import CenterCrop
from tensorflow.keras.layers import Rescaling

# Example image data, with values in the [0, 255] range
training_data = np.random.randint(0, 256, size=(64, 200, 200, 3)).astype("float32")

cropper = CenterCrop(height=150, width=150)
scaler = Rescaling(scale=1.0 / 255)

output_data = scaler(cropper(training_data))
print("shape:", output_data.shape)
print("min:", np.min(output_data))
print("max:", np.max(output_data))
```

```txt
shape: (64, 150, 150, 3)
min: 0.0
max: 1.0
```

## 使用 Keras 函数 API 构建模型

layer 是一种简单的输入-输出转换，例如前面介绍的 
