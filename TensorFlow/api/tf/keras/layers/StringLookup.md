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

该 layer 通过词汇查找将字符串转换为整数。该 layer 不会对输入字符串进行拆分或转换。如果需要拆分和 tokenize，建议使用 [TextVectorization](TextVectorization.md) layer。

该 layer 的词汇表可以在构造时提供，也可以通过 `adapt()` 学习。`adapt()` 将分析一个数据集，确定单个字符串 token 的频率，并使用这些 tokens 创建词汇表。如果词汇表大小有限制，则使用频率最高的 tokens，其它 tokens 视为 OOV（out-of-vocabulary）。



## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/layers/StringLookup
