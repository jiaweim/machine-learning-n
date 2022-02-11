# tf.data.Dataset

- [tf.data.Dataset](#tfdatadataset)
  - [方法](#方法)
    - [batch](#batch)
    - [cache](#cache)
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
