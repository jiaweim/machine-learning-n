# tf.data.Dataset

- [tf.data.Dataset](#tfdatadataset)
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

## 方法

### batch

```python
batch(
    batch_size, drop_remainder=False, num_parallel_calls=None, deterministic=None,
    name=None
)
```

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

### shuffle

```python
shuffle(
    buffer_size, seed=None, reshuffle_each_iteration=None, name=None
)
```

### take

```python
take(
    count, name=None
)
```

使用数据集最多 `count` 个元素创建一个 `Dataset`。

```python
>>> dataset = tf.data.Dataset.range(10)
>>> dataset = dataset.take(3)
>>> list(dataset.as_numpy_iterator())
[0, 1, 2]
```

- `count`

`tf.int64` 类型的标量 `tf.Tensor`，表示从该数据集中取出 `count` 个元素用来创建新数据集。如果 `count` 为 -1，或者 `count` 大于该数据集的 size，则新数据包含该数据集全部元素。

## 参考

- https://www.tensorflow.org/api_docs/python/tf/data/Dataset
