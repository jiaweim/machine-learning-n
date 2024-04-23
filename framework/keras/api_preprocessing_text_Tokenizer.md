# Tokenizer

- [Tokenizer](#tokenizer)
  - [简介](#简介)
  - [参数说明](#参数说明)
  - [方法](#方法)
    - [fit_on_sequences](#fit_on_sequences)
    - [fit_on_texts](#fit_on_texts)
    - [texts_to_sequences](#texts_to_sequences)

## 简介

文本标记工具类。

```python
tf.keras.preprocessing.text.Tokenizer(
    num_words=None,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True, split=' ', char_level=False, oov_token=None,
    document_count=0, **kwargs
)
```

通过将文本转换为整数序列（整数是字典中和 token 对应的索引）向量化文本语料库。

## 参数说明

- `num_words`

要保留的最大单词数，根据出现的频率选择单词，只保留最常见的 `num_words-1` 个单词。

- `filters`

字符串类型，对该字符串中包含的所有字符，都要从文本中提出。默认为所有标点符号，加上制表符、换行符，除去 ' 字符。

## 方法

### fit_on_sequences

```python
fit_on_sequences(
    sequences
)
```

根据 sequence 列表更新内部词汇表。

在使用 `sequence_to_matrix` 前需要调用该方法。

`sequences` 为 sequence 列表，而 `sequence` 是 word indices 列表。

### fit_on_texts

```python
fit_on_texts(
    texts
)
```

使用文本列表更新内部词汇表。

### texts_to_sequences

```python
texts_to_sequences(
    texts
)
```

将文本转换为整数序列。

- 只考虑频率最高的 `num_words-1`个单词；
- 只考虑 tokenizer 见过的单词。
