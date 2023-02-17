# TFRecord 和 tf.train.Example

- [TFRecord 和 tf.train.Example](#tfrecord-和-tftrainexample)
  - [简介](#简介)
  - [配置](#配置)
  - [tf.train.Example](#tftrainexample)
    - [tf.train.Example 数据类型](#tftrainexample-数据类型)
    - [创建 tf.train.Example](#创建-tftrainexample)
  - [参考](#参考)

***

## 简介

TFRecord 是一种用于存储 binary 数据的简单格式。

[Protocol buffers](https://protobuf.dev/) 是一个跨平台、跨语言的库，用于结构化数据的高效序列化。

Protocol 信息由 `.proto` 文件定义，这些文件通常是理解信息类型的最简单方法。

`tf.train.Example` 信息（或 protobuf）是一种灵活的信息类型，代表 `{"string": value}` 映射。用于 TF，并在所有高级 API（如 TFX）中使用。

下面演示如何创建、解析和使用 `tf.train.Example`，然后将其序列化并写入 `.tfrecord` 文件，已经重新读回 `tf.train.Example`。

> **NOTE**
> 一般来说，应该将数据拆分到多个文件，以便 I/O 并行化。根据经验，文件数至少是读取数据的主机数的 10 倍以上，同时每个文件要足够大（100+ MB），便于 I/O 预取。例如，假设有 X GB 数据，计划在 N 台主机上进行训练，理想情况下，应该有 ~10*N 个文件，每个文件 `~X/(10*N)` 应不小于 100 MB+。如果文件太小，可能需要创建更少的文件来平衡并行和 I/O 预期的优势。

## 配置

```python
import tensorflow as tf

import numpy as np
```

## tf.train.Example

### tf.train.Example 数据类型

`tf.train.Example` 本质上是 `{"string": tf.train.Feature}` 映射。

`tf.train.Feature` 接受三种类型，其它大多数类型都可以转换为其中一种：

1. `tf.train.BytesList` (支持以下类型)
   - string
   - byte

2. `tf.train.FloatList`（支持以下类型） 
   - float (float32)
   - double (float64)

3. `tf.train.Int64List`（支持以下类型）
    - bool
   - enum
   - int32
   - uint32
   - int64
   - uint64

为了将标准 TF 类型转换为 `tf.train.Feature`，可以使用下面的函数。每个函数输入标量，返回 `tf.train.Feature`：

```python
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
```

> **NOTE**
> 这里仅使用标准输入，处理非标量 features 的最简单方法是使用 `tf.io.serialize_tensor` 将张量转换为 binary-strings。string 在 TF 中为标量。然后用 `tf.io.parse_tensor` 将 binary-string 转换为张量。 

下面是这些函数的示例。追输入和输出类型。

```python
print(_bytes_feature(b'test_string'))
print(_bytes_feature(u'test_bytes'.encode('utf-8')))

print(_float_feature(np.exp(1)))

print(_int64_feature(True))
print(_int64_feature(1))
```

```txt
bytes_list {
  value: "test_string"
}

bytes_list {
  value: "test_bytes"
}

float_list {
  value: 2.7182817459106445
}

int64_list {
  value: 1
}

int64_list {
  value: 1
}
```

所有信息都可以使用 `.SerializeToString` 方法转换为 binary-string：

```python
feature = _float_feature(np.exp(1))

feature.SerializeToString()
```

```txt
b'\x12\x06\n\x04T\xf8-@'
```

### 创建 tf.train.Example



## 参考

- https://www.tensorflow.org/tutorials/load_data/tfrecord
