# Dataset

- [Dataset](#dataset)
  - [简介](#简介)
    - [源数据集](#源数据集)
    - [变换（Transformations）](#变换transformations)
    - [常用术语](#常用术语)
    - [支持类型](#支持类型)
  - [属性](#属性)
  - [方法](#方法)
    - [batch](#batch)
    - [cache](#cache)
    - [from_generator](#from_generator)
    - [from_tensor_slices](#from_tensor_slices)
    - [map](#map)
    - [padded_batch](#padded_batch)
    - [prefetch](#prefetch)
    - [repeat](#repeat)
    - [shuffle](#shuffle)
    - [take](#take)
  - [参考](#参考)

2022-01-18, 16:40
***

## 简介

```python
tf.data.Dataset(
    variant_tensor
)
```

表示元素集合。

`tf.data.Dataset` API 支持编写描述性的高效的输入改管道。`Dataset` 的使用遵循如下模式：

1. 从输入数据创建源数据集
2. 应用数据集转换（transformation）预处理数据
3. 迭代数据集并处理元素

迭代以流的方式进行，因此不需要将完整数据集载入内存。

### 源数据集

- 创建数据集的最简单方式是从 Python list 创建

```python
>>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
>>> for element in dataset:
>>>     print(element)
tf.Tensor(1, shape=(), dtype=int32)
tf.Tensor(2, shape=(), dtype=int32)
tf.Tensor(3, shape=(), dtype=int32)
```

- 一行一行地处理文件，用 `tf.data.TextLineDataset`

```python
dataset = tf.data.TextLineDataset(["file1.txt", "file2.txt"])
```

- 处理 `TFRecord` 格式，用 `TFRecordDataset`

```python
dataset = tf.data.TFRecordDataset(["file1.tfrecords", "file2.tfrecords"])
```

- 用文件名称满足特定 pattern 的所有文件创建数据集，用 `tf.data.Dataset.list_files`

```python
dataset = tf.data.Dataset.list_files("/path/*.txt")
```

### 变换（Transformations）

有了数据集后，可以使用变换来处理数据集：

```python
>>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
>>> dataset = dataset.map(lambda x: x * 2)
>>> list(dataset.as_numpy_iterator())
[2, 4, 6]
```

### 常用术语

**元素（Element）**

在数据集迭代器上调用 `next()` 返回的单个输出。元素可以是包含多个组件的嵌套结构。例如，元素 `(1, (3, "apple"))` 包含两个嵌套的 tuple。组件包括 `1`, `3` 和 `"apple"`。

**组件（Component）**

元素嵌套结构中的叶子。

### 支持类型

元素可以是 tuple, namedTuple 和 dict 的嵌套结构。需要注意的是，Python list 不是组件的嵌套结构，而是转换为张量，作为组件使用。例如，元素 `(1, [1, 2, 3])` 只有两个组件，张量 `1` 和张量 `[1, 2, 3]`。

元素组件可以是 `tf.TypeSpec` 表示的任何类型，包括 `tf.Tensor`, `tf.data.Dataset`, `tf.sparse.SparseTensor`, `tf.RaggedTensor` 和 `tf.TensorArray`。

```python
a = 1 # Integer element
b = 2.0 # Float element
c = (1, 2) # Tuple element with 2 components
d = {"a": (2, 2), "b": 3} # Dict element with 3 components
Point = collections.namedtuple("Point", ["x", "y"])
e = Point(1, 2) # Named tuple
f = tf.data.Dataset.range(10) # Dataset element
```

## 属性

**element_spec**

数据集元素的类型。

```python
>>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
>>> dataset.element_spec
TensorSpec(shape=(), dtype=tf.int32, name=None)
```

## 方法

### batch

Last updated: 2022-08-04, 16:35

```python
batch(
    batch_size,
    drop_remainder=False,
    num_parallel_calls=None,
    deterministic=None,
    name=None
)
```

将此数据集的连续元素组合成 batch。

|参数|说明|
|---|---|
|batch_size|`tf.int64` 类型标量，表示单个 batch 包含的元素个数|
|drop_remainder|(Optional) `tf.bool` 类型标量，当最后一个 batch 元素个数小于 `batch_size` 时是否删除该 batch，默认不删除|
|num_parallel_calls|(Optional) `tf.int64` 类型标量，并行异步计算的 batch 数。不指定就按顺序计算 batch；使用 `tf.data.AUTOTUNE` 则根据可用资源动态设置并行数|
|deterministic|(Optional)当指定 `num_parallel_calls`，该 boolean 用于指定转换生成元素的顺序。`False` 表示允许生成无序元素，转换速度更快。默认为 True|

例如：

```python
>>> dataset = tf.data.Dataset.range(8)
>>> dataset = dataset.batch(3)
>>> list(dataset.as_numpy_iterator())
[array([0, 1, 2], dtype=int64),
 array([3, 4, 5], dtype=int64),
 array([6, 7], dtype=int64)]
```

```python
>>> dataset = tf.data.Dataset.range(8)
>>> dataset = dataset.batch(3, drop_remainder=True)
>>> list(dataset.as_numpy_iterator())
[array([0, 1, 2], dtype=int64), array([3, 4, 5], dtype=int64)]
```

生成的元素多了一个大小为 `batch_size` 的维度（如果无法整除，最后一个元素为 `N % batch_size`）。如果你的程序依赖于具有相同外部维度的 batch，则应该将 `drop_remainder` 设置为 `True`，以避免产生较小的 batch。

### cache

Last updated: 2022-08-09, 13:06

```python
cache(
    filename='', name=None
)
```

- **filename**

`tf.string` 类型张量，用于指定缓存此数据集的目录。如果未指定 `filename`，数据集将缓存在内存中。

**返回**：`Dataset`

缓存该数据集中的元素。

在第一次迭代该数据集时（第一个 epoch），其元素将缓存到指定的文件或内存中。

> **Note:** 为了完成缓存，必须遍历整个数据集。否则，后续迭代将不使用缓存数据。

```python
>>> dataset = tf.data.Dataset.range(5)
>>> dataset = dataset.map(lambda x: x**2)
>>> dataset = dataset.cache()

>>> # 第一次迭代数据将使用 `range` 和 `map` 生成数据
>>> list(dataset.as_numpy_iterator())
[0, 1, 4, 9, 16]

>>> # 后续的迭代从缓存读取数据
>>> list(dataset.as_numpy_iterator())
[0, 1, 4, 9, 16]
```

当缓存到文件时，缓存的数据在运行期间持久保存。即使是第一次迭代数据，也是从缓存文件读取。在调用 `.cache` 之前更改输入管道无效，除非删除缓存文件或更改文件名。

```python
dataset = tf.data.Dataset.range(5)
dataset = dataset.cache("/path/to/file")
list(dataset.as_numpy_iterator())
# [0, 1, 2, 3, 4]
dataset = tf.data.Dataset.range(10)
dataset = dataset.cache("/path/to/file")  # Same file!
list(dataset.as_numpy_iterator())
# [0, 1, 2, 3, 4]
```

> **Note:** `cache` 使得每次迭代数据集产生完全相同的元素。如果希望实现随机迭代顺序，则应该在调用 `cache` 后调用 `shuffle`。

### from_generator

```python
@staticmethod
from_generator(
    generator,
    output_types=None,
    output_shapes=None,
    args=None,
    output_signature=None,
    name=None
)
```

使用 Python generator 创建 `Dataset`。

> [!WARNING]
> 有些参数已弃用：`(output_shapes, output_types)`。它们会在将来的版本删除，改用 `output_signature`。



### from_tensor_slices

```python
@staticmethod
from_tensor_slices(
    tensors, name=None
)
```

|参数|说明|
|---|---|
|tensors|A dataset element, whose components have the same first dimension. Supported values are documented here.|
|name|(Optional)操作名称|

用指定张量的切片创建 `Dataset`。

沿着指定张量的第一维进行切片。该操作保留输入张量的结构，删除每个张量的第一维，然后将所有子张量保存到数据集。所有输入张量的第一维的 size 必须相同。

- 1D 张量切片生成标量张量元素

```python
>>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
>>> list(dataset.as_numpy_iterator())
[1, 2, 3]
```

- 2D 张量切片生成 1D 张量元素

```python
>>> dataset = tf.data.Dataset.from_tensor_slices([[1, 2], [3, 4]])
>>> list(dataset.as_numpy_iterator())
[array([1, 2]), array([3, 4])]
```

- 对 1D 张量的元组切片生成包含标量张量的元组元素

```python
>>> dataset = tf.data.Dataset.from_tensor_slices(([1, 2], [3, 4], [5, 6]))
>>> list(dataset.as_numpy_iterator())
[(1, 3, 5), (2, 4, 6)]
```

- 保留字典结构

```python
>>> dataset = tf.data.Dataset.from_tensor_slices({"a": [1, 2], "b": [3, 4]})
>>> list(dataset.as_numpy_iterator()) == [{'a': 1, 'b': 3},
...                                       {'a': 2, 'b': 4}]
True
```

- 两个张量可以合并为一个 Dataset 对象

```python
>>> features = tf.constant([[1, 3], [2, 1], [3, 3]]) # ==> 3x2 tensor
>>> labels = tf.constant(['A', 'B', 'A']) # ==> 3x1 tensor
>>> dataset = Dataset.from_tensor_slices((features, labels))
```

- features 和 labels 张量也可以先转换为 Dataset 对象，然后合并

```python
>>> features_dataset = Dataset.from_tensor_slices(features)
>>> labels_dataset = Dataset.from_tensor_slices(labels)
>>> dataset = Dataset.zip((features_dataset, labels_dataset))
```

- 批量 feature 和 label 也可以用同样的方式转换为 `Dataset`

```python
>>> batched_features = tf.constant([[[1, 3], [2, 3]],
...                                 [[2, 1], [1, 2]],
...                                 [[3, 3], [3, 2]]], shape=(3, 2, 2))
>>> batched_labels = tf.constant([['A', 'A'],
...                               ['B', 'B'],
...                               ['A', 'B']], shape=(3, 2, 1))
>>> dataset = Dataset.from_tensor_slices((batched_features, batched_labels))
>>> for element in dataset.as_numpy_iterator():
...   print(element)
(array([[1, 3],
       [2, 3]]), array([[b'A'],
       [b'A']], dtype=object))
(array([[2, 1],
       [1, 2]]), array([[b'B'],
       [b'B']], dtype=object))
(array([[3, 3],
       [3, 2]]), array([[b'A'],
       [b'B']], dtype=object))
```

注意，如果 `tensors` 包含 NumPy 数组，且未启用 eager 执行，则这些值将以一个或多个 `tf.constant` 操作嵌入到 graph 中。对大型数据集（> 1GB），这会浪费内存，且可能超过 graph 序列化的大小限制。

### map

```python
map(
    map_func, num_parallel_calls=None, deterministic=None, name=None
)
```

### padded_batch

```python
padded_batch(
    batch_size,
    padded_shapes=None,
    padding_values=None,
    drop_remainder=False,
    name=None
)
```

|参数|说明|
|---|---|
|batch_size|

将数据集的元素合并为 padded batches。

次转换将输入数据集中多个连续元素合并为单个元素。

和 `tf.data.Dataset.batch` 一样，生成元素的组成外部多一个维度，长度为 `batch_size`（如果不能整除，最后一个 batch 为 `N % batch_size`）。如果需要所有 batch 元素个数相同，则应该设置 `drop_remainder=True`。

和 `tf.data.Dataset.batch` 不同的是，组合成 batch 的元素的 shape 可能不同，`padded_batch` 将元素的每个组件填充到指定 shape `padded_shapes`。`padded_shapes` 指定输出元素中每个组件的 shape:

- 如果指定的尺寸为常量，则组件在对应维度填充到该长度
- 如果尺寸未知，则组件会填充到所有元素该组件的最大长度

```python
A = tf.data.Dataset.range(1, 5, output_type=tf.int32)
                   .map(lambda x: tf.fill([x], x))
# Pad to the smallest per-batch size that fits all elements.
B = A.padded_batch(2)
for element in B.as_numpy_iterator():
    print(element)
```

### prefetch

Last updated: 2022-08-09, 13:13

```python
prefetch(
    buffer_size, name=None
)
```

从该数据集预先取一部分元素创建 `Dataset`。

**返回**：`Dataset`

大部分数据集输入管道应该以调用 `prefetch` 结尾。这样在处理当前元素时同时准备后续元素，从而提高吞吐量，降低延迟；代价是使用额外的内存来存储预取的元素。

> **Note：** 和其它 `Dataset` 方法一样，`prefetch` 对输入数据集的元素进行操作，没有样本和批量的概念。`examples.prefetch(2)` 将预取 2 个元素（即 2 个样本），而 `examples.batch(20).prefetch(2)` 虽然也是预取 2 个元素，但每个元素为 1 个批量，每个批量包含 20 个元素。

```python
>>> dataset = tf.data.Dataset.range(3)
>>> dataset = dataset.prefetch(2)
>>> list(dataset.as_numpy_iterator())
[0, 1, 2]
```

|参数|说明|
|---|---|
|buffer_size|`tf.int64` 标量 `tf.Tensor`, 表示预取时缓冲元素的最大个数。如果使用 `tf.data.AUTOTUNE`，则动态调整缓冲区大小|

### repeat

Last updated: 2022-08-04, 16:23

```python
repeat(
    count=None, name=None
)
```

重复该数据集 `count` 次。

```python
>>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
>>> dataset = dataset.repeat(3)
>>> list(dataset.as_numpy_iterator())
[1, 2, 3, 1, 2, 3, 1, 2, 3]
```

> [!NOTE]
> 如果输入数据集依赖于全局状态（如随机数生成器），或者其输出是不确定的（如上游的 shuffle 操作），则不同的重复可能产生不同的元素。

|参数|说明|
|---|---|
|count|（可选）`tf.int64` 类型的 `tf.Tensor` 标量, 表示重复数据集次数。默认（`count` 为 `None` 或 `-1`）无限重复|
|name|（可选）操作名称|

### shuffle

Last updated: 2022-06-15, 17:24

```python
shuffle(
    buffer_size, seed=None, reshuffle_each_iteration=None, name=None
)
```

随机打乱数据集的元素。

该数据集用 `buffer_size` 个元素填充缓冲区，然后从这个缓冲区中随机取样，再从数据集中取新的数据替换缓冲区中选中的元素。要实现完美洗牌，缓冲区大小不能小于数据集大小。

例如，如果数据集包含 10000 个元素，但是 `buffer_size` 设置为 1000，则 `shuffle` 首先从缓冲区的 1000 个元素中随机选择一个元素，缓冲区空出来的一个位置由第 1001 元素替换，从而保持缓冲区大小 1000 不变。

`reshuffle_each_iteration` 表示不同 epoch 的洗牌顺序是否不同。在 TF 1.X 中，通常用 `repeat` 转换创建 epochs：

```py
dataset = tf.data.Dataset.range(3)
dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
dataset = dataset.repeat(2)
# [1, 0, 2, 1, 2, 0]

dataset = tf.data.Dataset.range(3)
dataset = dataset.shuffle(3, reshuffle_each_iteration=False)
dataset = dataset.repeat(2)
# [1, 0, 2, 1, 0, 2]
```

在 TF 2.0 中 `tf.data.Dataset` 对象是 Python 可迭代对象，可以通过 Python 迭代创建 epoch：

```python
dataset = tf.data.Dataset.range(3)
dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
list(dataset.as_numpy_iterator())
# [1, 0, 2]
list(dataset.as_numpy_iterator())
# [1, 2, 0]
```

```python
dataset = tf.data.Dataset.range(3)
dataset = dataset.shuffle(3, reshuffle_each_iteration=False)
list(dataset.as_numpy_iterator())
# [1, 0, 2]
list(dataset.as_numpy_iterator())
# [1, 0, 2]
```

|参数|说明|
|---|---|
|`buffer_size`|`tf.int64` 类型的 `tf.Tensor` 标量，缓冲区大小|
|`seed`|(Optional.) `tf.int64` 类型 `tf.Tensor` 标量，表示随机 seed|
|reshuffle_each_iteration|(Optional.) boolean 值，表示每次迭代数据集时是否重新洗牌(Defaults to True.)|
|name|(Optional.) A name for the tf.data operation.|

### take

Last updated: 2022-08-04, 16:41

```python
take(
    count, name=None
)
```

使用数据集前 `count` 个元素创建一个新的 `Dataset`。即如果数据集包含的元素少于 `count`，则有多少用多少。

```python
>>> dataset = tf.data.Dataset.range(10)
>>> dataset = dataset.take(3)
>>> list(dataset.as_numpy_iterator())
[0, 1, 2]
```

|参数|说明|
|---|---|
|count|`tf.int64` 类型的标量，表示从该数据集中取出 `count` 个元素创建新数据集。如果 `count` 为 -1，或者 `count` 大于该数据集的 size，则新数据集包含该数据集全部元素|

## 参考

- https://www.tensorflow.org/api_docs/python/tf/data/Dataset
