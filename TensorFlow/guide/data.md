# tf.data：构建 TensorFlow 输入管道

- [tf.data：构建 TensorFlow 输入管道](#tfdata构建-tensorflow-输入管道)
  - [简介](#简介)
  - [基本原理](#基本原理)
    - [Dataset 结构](#dataset-结构)
  - [读取输入数据](#读取输入数据)
    - [NumPy 数组](#numpy-数组)
    - [Python generators](#python-generators)
  - [参考](#参考)

2022-01-18, 18:32
***

## 简介

使用 `tf.data` API 可以用简单的、可重复使用的代码片段构建复杂的输入管道。例如，用于图像模型的输入管道可以从分布式文件系统中的文件聚合数据，将随机扰动应用于每个图像，并将随机选择的图像合并为 batch 进行训练。文本模型的输入管道可能涉及从原始文本数据中提取符号，将它们通过查找表转换为嵌入，然后将不同长度的序列合并为 batch。`tf.data` API 提供了处理大型数据的功能，支持读取不同格式的数据，并执行复杂的转换。

`tf.data` API 引入了表示元素序列的 `tf.data.Dataset` 抽象类，每个元素由一个或多个组件组成。例如，在图像管道中，元素可能是单个样本，包含一对表示图像和其标签的张量。

创建数据集的方法有两种：

- 从内存或文件（一个或多个）中的数据构建 `Dataset`
- 从一个或多个 `tf.data.Dataset` 对象进行转换构建数据集。

```python
import tensorflow as tf
```

```python
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.set_printoptions(precision=4)
```

## 基本原理

要创建输入管道，必须提供数据源。例如，要从内存中的数据构造 `Dataset`，可使用 `tf.data.Dataset.from_tensors()` 或 `tf.data.Dataseet.from_tensor_slices()`；如果以 TFRecord 文件创建 `Dataset`，则可以使用 `tf.data.TFRecordDataset()`。

有了 `Dataset` 对象后，可以通过 `tf.data.Dataset` 的链式方法将其转换为新的 `Dataset`。例如，可以应用逐元素转换操作，如 `Dataset.map()`，也可以使用多元素转换功能，如 `Dataset.batch`。可使用的转换请参考 [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) API 文档。

`Dataset` 是可迭代 Python 对象，因此可以使用 for 循环来查看其元素：

```python
dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])
dataset
```

```txt
<TensorSliceDataset element_spec=TensorSpec(shape=(), dtype=tf.int32, name=None)>
```

```python
for elem in dataset:
    print(elem.numpy())
```

```txt
8
3
0
8
2
1
```

或者使用 `iter` 显式创建 Python 迭代器，然后用 `next` 查看元素：

```python
it = iter(dataset)

print(next(it).numpy())
```

```txt
8
```

另外，也可以使用 `reduce` 变换将所有元素转换为单个值。例如，使用 `reduce` 计算所有元素之和：

```python
print(dataset.reduce(0, lambda state, value: state + value).numpy())
```

```txt
22
```

### Dataset 结构

dataset 包含元素序列，每个元素具有相同的组件嵌套结构，而单个组件可以是 `tf.TypeSpec` 表示的任意类型，包括 `tf.Tensor`, `tf.sparse.SparseTensor`, `tf.RaggedTensor`, `tf.TensorArray` 和 `tf.data.Dataset`。

可用于表示嵌套结构的 Python 类型包含 `tuple`, `dict`, `NamedTuple` 和 `OrderedDict`。需要强调的是，`list` 不适合表示 dataset 元素，早期的 `tf.data` 用户对输入 `list` 被自动包装为 `tensor`，输出 `list` （如用户自定义函数）被强制转换为 `tuple` 反应比较强烈。因此，如果需要将 `list` 作为嵌套结构，则需要先将其转换为 `tuple`；如果需要将 `list` 作为单个组件输出，需要使用 `tf.stack` 进行包装。

`Dataset.element_spec` 属性可用于检查每个元素组件的类型。该属性返回 `tf.TypeSpec` 对象的嵌套结构，嵌套结构和元素结构一致。例如：

- 生成 shape 为 (4, 10) 的 [0,1) 之间的均匀分布随机数

```python
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))

dataset1.element_spec
```

```txt
TensorSpec(shape=(10,), dtype=tf.float32, name=None)
```

```python
dataset2 = tf.data.Dataset.from_tensor_slices(
    (tf.random.uniform([4]),
     tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))

dataset2.element_spec
```

```txt
(TensorSpec(shape=(), dtype=tf.float32, name=None),
 TensorSpec(shape=(100,), dtype=tf.int32, name=None))
```

```python
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

dataset3.element_spec
```

```txt
(TensorSpec(shape=(10,), dtype=tf.float32, name=None),
 (TensorSpec(shape=(), dtype=tf.float32, name=None),
  TensorSpec(shape=(100,), dtype=tf.int32, name=None)))
```

```python
# Dataset containing a sparse tensor.
>>> dataset4 = tf.data.Dataset.from_tensors(
>>>     tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]))
>>> dataset4.element_spec
SparseTensorSpec(TensorShape([3, 4]), tf.int32)
>>> dataset4.element_spec.value_type
tensorflow.python.framework.sparse_tensor.SparseTensor
```

`Dataset` transformations 支持任意结构的数据集。当使用 `Dataset.map()` 和 `Dataset.filter()` 进行转换时，元素结构决定了函数的参数：

```python
>>> dataset1 = tf.data.Dataset.from_tensor_slices(
>>>     tf.random.uniform([4, 10], minval=1, maxval=10, dtype=tf.int32))
>>> dataset1
<TensorSliceDataset element_spec=TensorSpec(shape=(10,), dtype=tf.int32, name=None)>
```

```python
>>> for z in dataset1:
>>>    print(z.numpy())
[3 8 8 4 6 2 3 5 8 5]
[9 3 2 1 5 1 1 2 4 3]
[1 1 4 2 3 2 2 8 8 1]
[5 3 6 4 6 4 1 9 1 1]
```

```python
>>> dataset2 = tf.data.Dataset.from_tensor_slices(
>>>     (tf.random.uniform([4]), tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))
>>> dataset2
<TensorSliceDataset element_spec=(TensorSpec(shape=(), dtype=tf.float32, name=None), TensorSpec(shape=(100,), dtype=tf.int32, name=None))>
```

```python
>>> dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
>>> dataset3
<ZipDataset element_spec=(TensorSpec(shape=(10,), dtype=tf.int32, name=None), (TensorSpec(shape=(), dtype=tf.float32, name=None), TensorSpec(shape=(100,), dtype=tf.int32, name=None)))>
>>> for a, (b, c) in dataset3:
>>>     print("shapes: {a.shape}, {b.shape}, {c.shape}".format(a=a, b=b, c=c))
shapes: (10,), (), (100,)
shapes: (10,), (), (100,)
shapes: (10,), (), (100,)
shapes: (10,), (), (100,)
```

## 读取输入数据

### NumPy 数组

更多示例请参考 [载入 NumPy 数组](../../tutorials/load_data/load_numpy.md)。

如果所有输入数据都在内存中，则创建 `Dataset` 的最简单方法是将它们转换为 `tf.Tensor` 对象，然后使用 `Dataset.from_tensor_slices()` 方法。

```python
train, test = tf.keras.datasets.fashion_mnist.load_data()
```

```sh
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
32768/29515 [=================================] - 0s 0us/step
40960/29515 [=========================================] - 0s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
26427392/26421880 [==============================] - 0s 0us/step
26435584/26421880 [==============================] - 0s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
16384/5148 [===============================================================================================] - 0s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
4423680/4422102 [==============================] - 0s 0us/step
4431872/4422102 [==============================] - 0s 0us/step
```

```python
images, labels = train
images = images/255

dataset = tf.data.Dataset.from_tensor_slices((images, labels))
dataset
```

```sh
<TensorSliceDataset element_spec=(TensorSpec(shape=(28, 28), dtype=tf.float64, name=None), TensorSpec(shape=(), dtype=tf.uint8, name=None))>
```

> 上面的代码将 features 和 labels 数组作为 `tf.constant()` 操作嵌入到 TensorFlow graph 中。这对于小型数据集有效，但是比价耗内存，因为数组的内容会被复制多次，并且可能超过 `tf.GraphDef` 协议缓存的 2GB 限制。

### Python generators

Python generator 是另一种可以很容易被 `tf.data.Dataset` 处理的数据源。

> 虽然 python generator 很便捷，但是可移植性和可扩展性有限。它必须在创建生成器的 Python 进程中运行，并且受 Python GIL 影响。

```python
def count(stop):
  i = 0
  while i<stop:
    yield i
    i += 1
```

```python
for n in count(5):
  print(n)
```

```sh
0
1
2
3
4
```

`Dataset.from_generator` 将 Python 生成器转换为 `tf.data.Dataset`。

`Dataset.from_generator` 方法以可调用对象作为输入，而不是迭代器。这样在迭代到末尾后，重新开始生成器。它包含一个可选参数 `args`，作为可调用对象的参数。

`output_types` 参数是必需的，因为 `tf.data` 在内部构建 `tf.Graph`，而 graph edges 需要 `tf.dtype`。

```python
ds_counter = tf.data.Dataset.from_generator(count, args=[25], output_types=tf.int32, output_shapes = (), )
```

```python
for count_batch in ds_counter.repeat().batch(10).take(10):
  print(count_batch.numpy())
```

```sh
[0 1 2 3 4 5 6 7 8 9]
[10 11 12 13 14 15 16 17 18 19]
[20 21 22 23 24  0  1  2  3  4]
[ 5  6  7  8  9 10 11 12 13 14]
[15 16 17 18 19 20 21 22 23 24]
[0 1 2 3 4 5 6 7 8 9]
[10 11 12 13 14 15 16 17 18 19]
[20 21 22 23 24  0  1  2  3  4]
[ 5  6  7  8  9 10 11 12 13 14]
[15 16 17 18 19 20 21 22 23 24]
```

## 参考

- https://www.tensorflow.org/guide/data
