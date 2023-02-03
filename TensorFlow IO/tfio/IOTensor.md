# tfio.IOTensor

- [tfio.IOTensor](#tfioiotensor)
  - [简介](#简介)
  - [Indexable vs. Iterable](#indexable-vs-iterable)
  - [方法](#方法)
    - [from\_hdf5](#from_hdf5)
  - [参考](#参考)

***

## 简介

```python
tfio.IOTensor(
    spec, internal=False
)
```

`IOTensor` 是由 IO 提供数据支持的张量。例如，`AudioIOTensor` 指数据来自音频文件的张量，`KafkaIOTensor` 指数据来自 Kafka 数据流服务器的张量。

`IOTensor` 可索引，支持 `__getitem__()` 和 `__len__()` 方法。换句话说，它是 `collections.abc.Sequence` 的子类。

示例：

```python
>>> import tensorflow_io as tfio
>>>
>>> samples = tfio.IOTensor.from_audio("sample.wav")
>>> print(samples[1000:1005])
... tf.Tensor(
... [[-3]
...  [-7]
...  [-6]
...  [-6]
...  [-5]], shape=(5, 1), dtype=int16)
```



## Indexable vs. Iterable


## 方法

### from_hdf5

```python
@classmethod
from_hdf5(
    filename, spec=None, **kwargs
)
```

从 HDF5 文件创建 `IOTensor`。


## 参考

- https://www.tensorflow.org/io/api_docs/python/tfio/IOTensor
