# tf.keras.layers.StringLookup

2022-03-04, 22:13
***

## 简介

```python
tf.keras.layers.StringLookup(
    max_tokens=None, num_oov_indices=1, mask_token=None,
    oov_token='[UNK]', vocabulary=None, idf_weights=None, encoding=None,
    invert=False, output_mode='int', sparse=False,
    pad_to_max_tokens=False, **kwargs
)
```

将字符串特征映射为整数索引的预处理层。

该 layer 通过基于表格的词汇查找将字符串转换为整数。该 layer 不会对输入字符串进行拆分或转换。如果需要拆分和 tokenize，请参考 [TextVectorization](TextVectorization.md) layer。

该 layer 的词汇表可以在构造时提供，也可以通过 `adapt()` 学习。

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/layers/StringLookup
