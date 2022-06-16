# tf.data.Dataset

- [tf.data.Dataset](#tfdatadataset)
  - [简介](#简介)
  - [方法](#方法)
    - [batch](#batch)
    - [cache](#cache)
    - [from_tensor_slices](#from_tensor_slices)
    - [map](#map)
    - [prefetch](#prefetch)
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




## 方法

### batch

```python
batch(
    batch_size, drop_remainder=False, num_parallel_calls=None, deterministic=None,
    name=None
)
```

将此数据集的元素进行组合成批量，返回新的 `Dataset`。例如：

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

生成的元素多了一个大小为 `batch_size` 的维度（如果无法整除，最后一个元素为 `N % batch_size`）。如果你的程序依赖于具有相同外部维度的批处理，则应该将 `drop_remainder` 设置为 `True`，以避免产生较小的 batch。

|参数|说明|
|---|---|
|batch_size|[tf.int64](../tf.md) 标量 [tf.Tensor](../Tensor.md)，合并成单个 batch 的连续元素数目|
|drop_remainder|(Optional.) [tf.bool](../tf.md) 标量 [tf.Tensor](../Tensor.md)，当最后一个 batch 的元素个数小于 `batch_size` 时是否删除该 batch，默认不删除|
|num_parallel_calls|(Optional.) [tf.int64](../tf.md) 标量 [tf.Tensor](../Tensor.md)，并行异步计算的 batch 数。如果未指定，则按顺序计算。如果使用 [tf.data.AUTOTUNE](tf.data.md)，则根据可用资源动态设置|
|deterministic|(Optional.)当指定 `num_parallel_calls`，该 boolean 用于指定转换生成元素的顺序。如果为 `False`，则允许生成无序元素，牺牲确定性换区性能。默认为 True。|
|name|(Optional.) A name for the tf.data operation.|


### cache

```python
cache(
    filename='', name=None
)
```

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
|name|(Optional.) A name for the tf.data operation.|

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

注意，如果 `tensors` 包含 NumPy 数组，并且未启用 eager 执行，则这些值将以一个或多个 `tf.constant` 操作嵌入到 graph 中。对大型数据集（> 1GB），这回浪费内存。



### map

```python
map(
    map_func, num_parallel_calls=None, deterministic=None, name=None
)
```

### prefetch

```python
prefetch(
    buffer_size, name=None
)
```

从该数据集预先取一部分元素创建 `Dataset`，返回 `Dataset`。

大部分数据集输入管道应该以调用 `prefetch` 结尾。这样在处理当前元素时同时在准备后面的元素，从而提高吞吐量，降低延迟，但单价是使用额外的内存来存储预取的元素。

> ⭐：和其它 `Dataset` 方法一样，`prefetch` 对输入数据集的元素进行操作，没有样本和批量的概念。`examples.prefetch(2)` 将预取 2 个元素（即 2 个样本），而 `examples.batch(20).prefetch(2)` 虽然也是预取 2 个元素，但每个元素为 1 个批量，每个批量包含 20 个元素。

```python
>>> dataset = tf.data.Dataset.range(3)
>>> dataset = dataset.prefetch(2)
>>> list(dataset.as_numpy_iterator())
[0, 1, 2]
```

|参数|说明|
|---|---|
|buffer_size|`tf.int64` 类型的标量 `tf.Tensor`, 表示预取时缓冲元素的最大个数。如果使用 `tf.data.AUTOTUNE`，则动态调整缓冲区大小|
|name|Optional. A name for the tf.data transformation.|

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

```python
take(count, name=None)
```

使用数据集至多 `count` 个元素创建一个 `Dataset`。即如果数据集包含的元素个数少于 `count`，则有多少用多少。

```python
>>> dataset = tf.data.Dataset.range(10)
>>> dataset = dataset.take(3)
>>> list(dataset.as_numpy_iterator())
[0, 1, 2]
```

|参数|说明|
|---|---|
|count|`tf.int64` 类型的标量 `tf.Tensor`，表示从该数据集中取出 `count` 个元素创建新数据集。如果 `count` 为 -1，或者 `count` 大于该数据集的 size，则新数据包含该数据集全部元素。|
|name|(Optional.) A name for the tf.data operation.|

## 参考

- https://www.tensorflow.org/api_docs/python/tf/data/Dataset
