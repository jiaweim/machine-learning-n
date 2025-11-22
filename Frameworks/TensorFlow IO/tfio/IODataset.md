# tfio.IODataset

- [tfio.IODataset](#tfioiodataset)
  - [简介](#简介)
  - [参数](#参数)
  - [属性](#属性)
  - [方法](#方法)
    - [apply](#apply)
    - [batch](#batch)
    - [from\_hdf5](#from_hdf5)
  - [参考](#参考)

Last updated: 2023-02-01, 10:40
****

## 简介

```python
tfio.IODataset(
    function, internal=False, **kwargs
)
```

`IODataset` 是 `tf.data.Dataset` 的子类，是支持 IO 的数据类。`IODataset` 是确定性的，即数据有限且可重复。`IODataset` 可以直接传入 `tf.keras` 用于训练或预测。

虽然 `IODataset` 是 `tf.data.Dataset` 的子类，但并非所有 `tf.data.Dataset` 都是确定性的。`tf.data.Dataset` 也可以是 `StreamIODataset`，持续不断的生成数据且不可重复。`StreamIODataset` 只能传递给 `tf.keras` 用来预测。

`IODataset` 可用于音频数据文件（如 WAV 文件）的数据集 `AudioIODataset`。`KafkaIODataset` 也是一个 `IODataset`，但是能反复运行，每次生成相同数据。

例如：

```python
import tensorflow as tf
import tensorflow_io as tfio

audio = tfio.IODataset.from_audio("sample.wav")
```

## 参数

- **variant_tensor**

一个 `DT_VARIANT` 张量，用于表示 dataset。

## 属性

- **element_spec**

数据集数据类型规范。例如：

```python
>>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
>>> dataset.element_spec
TensorSpec(shape=(), dtype=tf.int32, name=None)
```

## 方法

### apply

```python
apply(
    transformation_func
)
```

应用转换函数。



### batch

```python
batch(
    batch_size, 
    drop_remainder=False, 
    num_parallel_calls=None, deterministic=None,
    name=None
)
```

### from_hdf5

```python
@classmethod
from_hdf5(
    filename, dataset, spec=None, **kwargs
)
```

使用 hdf5 文件的 dataset 对象创建 `IODataset`。

**参数：**

- **filename** - string，hdf5 文件名。

- **dataset** - string, hdf5 文件中的 dataset 名称，注意，HDF5 数据集名称以 `/` 开头。

- **spec**

dataset 的 `tf.TensorSpec` 或 dtype (如 `tf.int64`)。graph 模式需要 spec。在 eager 模式，自动检测 `spec`。

- **name**

`IOTensor` 名称前缀（可选）。

**返回：**

`IODataset`。

## 参考

- https://www.tensorflow.org/io/api_docs/python/tfio/IODataset
