# 载入文本

## 简介

下面介绍加载和预处理文本的两种方法：

- 使用 keras 工具和预处理层。包括将数据转换为 `tf.data.Dataset` 的 `tf.keras.utils.text_dataset_from_directory` 以及用于数据标准化、标记化和矢量化的 `tf.keras.layers.TextVectorization`。对新手，建议使用该方法。
- 使用底层 API，如 `tf.data.TextLineDataset` 加载文本文件，用 TensorFlow Text APIs 如 `text.UnicodeScriptTokenizer` 和 `text.case_fold_utf8` 等预处理数据。


## 参考

- https://www.tensorflow.org/tutorials/load_data/text
