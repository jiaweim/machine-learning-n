# StringLookup

2022-03-04, 22:13
@author Jiawei Mao
***

## 简介

```python
tf.keras.layers.StringLookup(
    max_tokens=None,
    num_oov_indices=1,
    mask_token=None,
    oov_token='[UNK]',
    vocabulary=None,
    idf_weights=None,
    encoding=None,
    invert=False,
    output_mode='int',
    sparse=False,
    pad_to_max_tokens=False,
    **kwargs
)
```

将字符串特征映射为整数索引的预处理层。

该 layer 通过词汇查找表将字符串转换为整数。该 layer 不会对输入字符串进行拆分或转换。如果需要拆分和 tokenize，建议使用 `TextVectorization` layer。

该 layer 的词汇表可以在构造时提供，也可以通过 `adapt()` 学习。`adapt()` 将分析数据集，确定单个字符串 token 的频率，并使用这些 tokens 创建词汇表。如果词汇表大小有限制，则使用频率最高的 tokens 创建词汇表，其它 tokens 视为 OOV（out-of-vocabulary）。

`StringLookup` 有两种输出模式：

- `output_mode` 为 `int` 时，输入字符串转换为它们在词汇表的索引
- `output_mode` 为 `multi_hot`, `count` 或 `tf_idf` 时，输入字符串转换为数组，数组大小与词汇表大小一致

词汇表可以包含 mask token 和 OOV token，OOV token 可以占用多个索引，使用 `num_oov_indices` 设置。这些 tokens 在词汇表中的位置是固定的：

- 当 `output_mode` 为 `int`，词汇表以 mask token 开始（如果有），然后为 OOV 索引，最后是其余词汇；
- 当 `output_mode` 为 `multi_hot`, `count` 或 `tf_idf`，词汇表以 OOV 索引开始，而 mask token 被删除。

## 参数

|参数|说明|
|---|---|
|max_tokens|词汇表的最大 size。只能在 adapt 词汇表或设置 `pad_to_max_tokens=True` 时指定。`None` 表示不限制词汇表大小。注意，该 size 包含 OOV 和 mask tokens。默认 `None`|
|num_oov_indices|OOV tokens 数。大于 1 时，使用 OOV 输入的 hash 值确定 OOV 值。为 0 时，OOV 输入会导致错误。默认为 1|
|mask_token|表示 mask 输入的 token。`output_mode` 为 `int` 时，该 token 包含在词汇表中，索引为 0。在其它输出模式中，词汇表不包括 mask token，输入中的 mask token 直接删除。`None` 则不设置 mask。默认 `None`|
|oov_token|Only used when invert is True. The token to return for OOV indices. Defaults to "[UNK]".
|vocabulary|可选。字符串数组或文本文件路径。字符串数组支持 tuple, list, 1D numpy array 以及 1D tensor 的字符串类型；如果文本文件路径，则要求文件的每行包含词汇表的一项。设置该参数后，无需调用 `adapt()`|
|idf_weights|Only valid when output_mode is "tf_idf". A tuple, list, 1D numpy array, or 1D tensor or the same length as the vocabulary, containing the floating point inverse document frequency weights, which will be multiplied by per sample term counts for the final tf_idf weight. If the vocabulary argument is set, and output_mode is "tf_idf", this argument must be supplied.
|invert|`output_mode` 为 "int" 才有效。True 时将索引映射为词汇表项，默认 False|
|output_mode|输出模式。默认 "int"，可选值 "int", "one_hot", "multi_hot", "count" 和"tf_idf"|
|pad_to_max_tokens|Only applicable when output_mode is "multi_hot", "count", or "tf_idf". If True, the output will have its feature axis padded to max_tokens even if the number of unique tokens in the vocabulary is less than max_tokens, resulting in a tensor of shape [batch_size, max_tokens] regardless of vocabulary size. Defaults to False.
|sparse|Boolean. Only applicable when output_mode is "multi_hot", "count", or "tf_idf". If True, returns a SparseTensor instead of a dense Tensor. Defaults to False.

输出模式：

- "int"，返回输入 tokens 的原始整数索引
- "one_hot"，独热编码，即输入的每个元素由一个数组表示，数组长度与词汇表长度相同，在对应元素位置为 1。如果最后一个维度大小为 1，则在该维度上进行编码；如果不是 1，则添加一个新维度。
- "multi_hot"，将输入的每个样本编码为一个数组，数组长度与词汇表长度相同，对样本存在元素的位置为 1。将最后一个维度视为样本维度，如果输入 shape 为 `(..., sample_length)`，则输出 shape 为 `(..., num_tokens)`。
- "count"，同 "multi_hot"，但是输出数组在各个元素位置处为样本中该元素出现的个数，而不是 1。
- `tf_idf`，同 "multi_hot"，但是使用 TF-IDF 算法查找每个 token 的值。

## 示例

### 使用已有词汇表创建 lookup

使用已有词汇表创建一个 `StringLookup` 层：

```python
>>> vocab = ["a", "b", "c", "d"]
>>> data = tf.constant([["a", "c", "d"], ["d", "z", "b"]])
>>> layer = tf.keras.layers.StringLookup(vocabulary=vocab)
>>> layer(data)
<tf.Tensor: shape=(2, 3), dtype=int64, numpy=
array([[1, 3, 4],
       [4, 0, 2]], dtype=int64)>
```

默认为 `int` 模式，由于没有 mask token，所以 OOV 索引为 0。

### 通过 adapt 词汇表创建 lookup

通过分析数据集创建词汇表：

```python
>>> data = tf.constant([["a", "c", "d"], ["d", "z", "b"]])
>>> layer = tf.keras.layers.StringLookup()
>>> layer.adapt(data)
>>> layer.get_vocabulary()
['[UNK]', 'd', 'z', 'c', 'b', 'a']
```

可以看到，OOV token '[UNK]' 也添加到了词汇表。余下的 tokens 按照频率排序，`"d"` 出现 2 次，所以排第一。

```python
>>> data = tf.constant([["a", "c", "d"], ["d", "z", "b"]])
>>> layer = tf.keras.layers.StringLookup()
>>> layer.adapt(data)
>>> layer(data)
<tf.Tensor: shape=(2, 3), dtype=int64, numpy=
array([[5, 3, 1],
       [1, 2, 4]], dtype=int64)
```

### 多个 OOV 索引

使用包含多个 OOV 索引的 `StringLookup` 层。当创建包含多个 OOV 索引的 `StringLookup` 时，OOV 值根据 hash 值分布到不同 OOV 索引中，以确定的方式在整个集合中分布 OOV 值。

```python
>>> vocab = ["a", "b", "c", "d"]
>>> data = tf.constant([["a", "c", "d"], ["m", "z", "b"]])
>>> layer = tf.keras.layers.StringLookup(vocabulary=vocab, num_oov_indices=2)
>>> layer(data)
<tf.Tensor: shape=(2, 3), dtype=int64, numpy=
array([[2, 4, 5],
       [0, 1, 3]], dtype=int64)>
```

注意，OOV 值 'm' 的输出为 0，而 OOV 值 'z' 的输出为 1。为了给额外的 OOV 值腾出位置，词汇表范围内的索引都向后移动了 1 位。

## 方法

### get_vocabulary

```python
get_vocabulary(
    include_special_tokens=True
)
```

返回当前词汇表。

`include_special_tokens` 为 `True` 时返回的词汇表包含 mask 和 OOV  tokens，

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/layers/StringLookup
