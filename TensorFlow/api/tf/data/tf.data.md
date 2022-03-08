# tf.data

2022-03-03, 00:00
****

## 简介

`tf.data.Dataset` API 用于构建输入数据。`tf.data` API 提供了处理大规模数据的功能，并支持读取多种格式的数据。

`tf.data.Dataset` 提供了对序列数据的抽象，序列中的每个元素可以包含多个组成。例如对图像数据，一个元素可能就是一个训练样本，包含图像和标签一对 tensor。

`tf.data` 提供了两种创建数据集的方式：

- 从文件或内存中的数据构建的 `Dataset` 
- 从一个或多个 `tf.data.Dataset` 对象转换得到的数据集

## 类

|类|说明|
|---|---|
|[Dataset](Dataset.md)|Represents a potentially large set of elements.|
|DatasetSpec|Type specification for tf.data.Dataset.|
|FixedLengthRecordDataset|A Dataset of fixed-length records from one or more binary files.|
|Iterator|Represents an iterator of a tf.data.Dataset.|
|IteratorSpec|Type specification for tf.data.Iterator.|
|Options|Represents options for tf.data.Dataset.|
|TFRecordDataset|A Dataset comprising records from one or more TFRecord files.|
|TextLineDataset|Creates a Dataset comprising lines from one or more text files.|
|ThreadingOptions|Represents options for dataset threading.|

## 参考

- https://www.tensorflow.org/api_docs/python/tf/data
