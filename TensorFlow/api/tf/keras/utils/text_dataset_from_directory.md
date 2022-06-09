# text_dataset_from_directory

- [text_dataset_from_directory](#text_dataset_from_directory)
  - [简介](#简介)
  - [参数](#参数)
  - [返回值](#返回值)
  - [参考](#参考)

2022-03-04, 00:44
@author Jiawei Mao
****

## 简介

```python
tf.keras.utils.text_dataset_from_directory(
    directory,
    labels='inferred',
    label_mode='int',
    class_names=None,
    batch_size=32,
    max_length=None,
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    follow_links=False
)
```

使用目录中的文本文件生成 [tf.data.Dataset](../../data/Dataset.md)。

如果目录结构为：

```sh
main_directory/
...class_a/
......a_text_1.txt
......a_text_2.txt
...class_b/
......b_text_1.txt
......b_text_2.txt
```

则调用 `text_dataset_from_directory(main_directory, labels='inferred')` 返回一个 [tf.data.Dataset](../../data/Dataset.md)，它从子目录 `class_a` 和 `class_b` 生成批量文本和对应标签（0 对应 `class_a`，1 对应 `class_b`）。

目前只支持 `.txt` 文件。

## 参数

|参数|说明|
|---|---|
|directory|数据所在目录。如果 `labels` 为 "inferred"，则该目录应该包含子目录，每个子目录包含一个类别的文件，否则忽略目录结构。|
|labels|指定标签|
|label_mode|标签模式|
|class_names|"labels" 为 "inferred" 生效。类别名称的显式列表，必须和子目录名称匹配。用于控制类的顺序，否则默认使用字母数字顺序|
|batch_size|批量大小，默认 32。`None` 表示不对数据进行批处理，每次生成单个样本|
|max_length|文本字符串的最大 size。超过 `max_length` 的文本被截断到 `max_length`|
|shuffle|是否打乱数据，默认 `True`。`False` 时则按字母数字顺序对数据排序|
|seed|用于打乱和转换的随机种子（可选）|
|validation_split|留作验证的数据比例，0 到 1 之间的浮点数（可选）|
|subset|"training" or "validation"。仅在设置 `validation_split` 时使用|
|follow_links|是否访问符号链接指向的子目录，默认 False。|

**labels** 取值如下：

- "inferred"，根据目录结构生成标签，默认选项；
- None，无标签；
- 和目录中文件数相同的整数标签 list/tuple，顺序要与文本文件路径的 alphanumeric 顺序一致，可以使用 `os.walk(directory)` 获得文本文件路径顺序。

**label_mode** 取值如下：

- 'int'，标签编码为整数，对应 `sparse_categorical_crossentropy` 损失函数，默认选项；
- 'categorical'，标签编码为分类向量，对应 `categorical_crossentropy` 损失函数；
- 'binary'，标签只有两个，被编码为 0 或 1 的 float32 标量，对应 `binary_crossentropy` 损失函数；
- None，无标签。

## 返回值

返回 `tf.data.Dataset` 对象：

- 如果 `label_mode` 为 None，生成 shape 为 `(batch_size, )` 的 string 张量，包含批量文本文件；
- 否则生成 tuple `(texts, labels)`，其中 `texts` shape 为 `(batch_size,)`, `labels` 格式如下。

labels 格式说明：

- 如果 `label_mode` 为 `int`，则标签是 shape 为 `(batch_size,)` 的 `int32` 张量；
- 如果 `label_mode` 为 `binary`，则标签是 shape 为 `(batch_size, 1)` 的 `float32` 张量，值为 0s 或 1s；
- 如果 `label_mode` 为 `categorial`，则标签是 shape 为 `(batch_size, num_classes)` 的 `float32` 张量，表示类别索引的 one-hot 编码。

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/utils/text_dataset_from_directory
