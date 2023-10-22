# Unicode 字符串

- [Unicode 字符串](#unicode-字符串)
  - [1. 简介](#1-简介)
  - [2. tf.string 数据类型](#2-tfstring-数据类型)
  - [3. 表示 Unicode](#3-表示-unicode)
    - [3.1 不同表示之间的转换](#31-不同表示之间的转换)
    - [3.2 batch 维度](#32-batch-维度)
  - [4. Unicode 操作](#4-unicode-操作)
    - [4.1 字符长度](#41-字符长度)
    - [4.2 子字符串](#42-子字符串)
    - [4.3 拆分 Unicode 字符串](#43-拆分-unicode-字符串)
    - [4.4 Byte offsets for characters](#44-byte-offsets-for-characters)
  - [5. Unicode scripts](#5-unicode-scripts)
  - [6. 示例：简单分词](#6-示例简单分词)
  - [7. 参考](#7-参考)

Last updated: 2022-08-08, 17:00
@author Jiawei Mao
****

## 1. 简介

NLP 模型通常需要处理不同语言的不同字符集。Unicode 是一个标准编码系统，包含几乎所有语言的字符。每个 Unicode 字符使用 `0` 到 `0x10FFFF` 之间的唯一整数代码点进行编码，一条 Unicode 字符串是 0 或多个 unicode 代码点的序列。

本教程演示如何在 TF 中表示 Unicode 字符串，并使用与标准字符串操作等价的 Unicode 操作进行处理。基于 script 检测拆分 Unicode 字符串为 tokens。

```python
import tensorflow as tf
import numpy as np
```

## 2. tf.string 数据类型

可以使用基本的 `tf.string` dtype 构建 byte 字符串张量。Unicode 字符串默认使用 [utf-8](https://en.wikipedia.org/wiki/UTF-8) 编码。

```python
tf.constant(u"Thanks 😊")
```

> [!NOTE]
> 因为 UTF-8 是默认编码，所以前面的 `u` 可以省略，这里为了清晰而加上，下同。

```txt
<tf.Tensor: shape=(), dtype=string, numpy=b'Thanks \xf0\x9f\x98\x8a'>
```

`tf.string` 张量将 byte 字符串视为原子单元，因此可以存储不同长度的 byte 字符串。字符串长度不包含在张量 shape 中。

```python
tf.constant([u"You're", u"welcome!"]).shape
```

```txt
TensorShape([2])
```

如果使用 Python 构造字符串，字符串字面量默认为 Unicode 编码。

## 3. 表示 Unicode

在 TensorFlow 中有两种表示 Unicode 字符串的标准方法：

- `string` 标量，代码点序列使用已知[字符编码](https://en.wikipedia.org/wiki/Character_encoding)进行编码。
- `int32` 向量，每个位置包含一个代码点。

例如，下面三个值都表示 Unicode 字符串 `"语言处理"`。

1. UTF-8 编码的 string 标量

```python
# Unicode string, UTF-8 编码的 string 标量
text_utf8 = tf.constant(u"语言处理")
text_utf8
```

```txt
<tf.Tensor: shape=(), dtype=string, numpy=b'\xe8\xaf\xad\xe8\xa8\x80\xe5\xa4\x84\xe7\x90\x86'>
```

2. UTF-16-BE 编码的 string 标量

```python
# Unicode string, UTF-16-BE 编码的 string 标量
text_utf16be = tf.constant(u"语言处理".encode("UTF-16-BE"))
text_utf16be
```

```txt
<tf.Tensor: shape=(), dtype=string, numpy=b'\x8b\xed\x8a\x00Y\x04t\x06'>
```

3. Unicode 代码点向量表示的 Unicode 字符串

```python
# Unicode string, 表示为 Unicode 代码点向量
text_chars = tf.constant([ord(char) for char in u"语言处理"])
text_chars
```

```txt
<tf.Tensor: shape=(4,), dtype=int32, numpy=array([35821, 35328, 22788, 29702])>
```

### 3.1 不同表示之间的转换

TF 提供了这些不同表示之间的转换功能：

- `tf.strings.unicode_decode`：将编码 string 标量转换为代码点向量
- `tf.strings.unicode_encode`：将代码点向量转换为编码 string 标量
- `tf.strings.unicode_transcode`：将编码 string 标量转换为不同编码

1. string 标量转换为代码点向量

```python
tf.strings.unicode_decode(text_utf8, input_encoding="UTF-8")
```

```txt
<tf.Tensor: shape=(4,), dtype=int32, numpy=array([35821, 35328, 22788, 29702])>
```

2. 代码点向量转换为 string 标量

```python
tf.strings.unicode_encode(text_chars, output_encoding="UTF-8")
```

```txt
<tf.Tensor: shape=(), dtype=string, numpy=b'\xe8\xaf\xad\xe8\xa8\x80\xe5\xa4\x84\xe7\x90\x86'>
```

3. utf-8 编码转换为 utf-16-be 编码

```python
tf.strings.unicode_transcode(
    text_utf8, input_encoding="UTF8", output_encoding="UTF-16-BE"
)
```

```txt
<tf.Tensor: shape=(), dtype=string, numpy=b'\x8b\xed\x8a\x00Y\x04t\x06'>
```

### 3.2 batch 维度

当解码多个字符串时，由于不同字符串中的字符数可能不同，因此返回 `tf.RaggedTensor` 类型。最里面维度的长度取决于最长字符串包含的字符数。

```python
# 一批 utf-8 编码的 Unicode 字符串
batch_utf8 = [
    s.encode("UTF-8")
    for s in [u"hÃllo", u"What is the weather tomorrow", u"Göödnight", u"😊"]
]
# 转换为 unicode 代码点
batch_chars_ragged = tf.strings.unicode_decode(batch_utf8, input_encoding="UTF-8")
for sentence_chars in batch_chars_ragged.to_list():
    print(sentence_chars)
```

```txt
[104, 195, 108, 108, 111]
[87, 104, 97, 116, 32, 105, 115, 32, 116, 104, 101, 32, 119, 101, 97, 116, 104, 101, 114, 32, 116, 111, 109, 111, 114, 114, 111, 119]
[71, 246, 246, 100, 110, 105, 103, 104, 116]
[128522]
```

返回的 `tf.RaggedTensor` 可以直接使用，也可以使用 `tf.RaggedTensor.to_tensor` 通过填充转换为 `tf.Tensor`（密集张量），或者使用 `tf.RaggedTensor.to_sparse` 转换为 `tf.sparsee.SparseTensor`（稀疏张量）。

```python
# 转换为密集张量
batch_chars_padded = batch_chars_ragged.to_tensor(default_value=-1)
print(batch_chars_padded.numpy())
```

```txt
[[   104    195    108    108    111     -1     -1     -1     -1     -1
      -1     -1     -1     -1     -1     -1     -1     -1     -1     -1
      -1     -1     -1     -1     -1     -1     -1     -1]
 [    87    104     97    116     32    105    115     32    116    104
     101     32    119    101     97    116    104    101    114     32
     116    111    109    111    114    114    111    119]
 [    71    246    246    100    110    105    103    104    116     -1
      -1     -1     -1     -1     -1     -1     -1     -1     -1     -1
      -1     -1     -1     -1     -1     -1     -1     -1]
 [128522     -1     -1     -1     -1     -1     -1     -1     -1     -1
      -1     -1     -1     -1     -1     -1     -1     -1     -1     -1
      -1     -1     -1     -1     -1     -1     -1     -1]]
```

```python
# 转换为稀疏张量
batch_chars_sparse = batch_chars_ragged.to_sparse()

nrows, ncols = batch_chars_sparse.dense_shape.numpy()
elements = [["_" for i in range(ncols)] for j in range(nrows)]
for (row, col), value in zip(
    batch_chars_sparse.indices.numpy(), batch_chars_sparse.values.numpy()
):
    elements[row][col] = str(value)
# max_width = max(len(value) for row in elements for value in row)
value_lengths = []
for row in elements:
    for value in row:
        value_lengths.append(len(value))
max_width = max(value_lengths)
print(
    "[%s]" % "\n ".join(
        "[%s]" % ", ".join(value.rjust(max_width) for value in row) for row in elements
    )
)
```

```txt
[[   104,    195,    108,    108,    111,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _]
 [    87,    104,     97,    116,     32,    105,    115,     32,    116,    104,    101,     32,    119,    101,     97,    116,    104,    101,    114,     32,    116,    111,    109,    111,    114,    114,    111,    119]
 [    71,    246,    246,    100,    110,    105,    103,    104,    116,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _]
 [128522,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _,      _]]
```

- 编码多个长度相同的字符串时，使用 `tf.Tensor` 作为输入

```python
tf.strings.unicode_encode(
    [[99, 97, 116], [100, 111, 103], [99, 111, 119]], output_encoding="UTF-8"
)
```

```txt
<tf.Tensor: shape=(3,), dtype=string, numpy=array([b'cat', b'dog', b'cow'], dtype=object)>
```

- 编码多个长度不同的字符串时，使用 `tf.RaggedTensor` 作为输入

```python
tf.strings.unicode_encode(batch_chars_ragged, output_encoding="UTF-8")
```

```txt
<tf.Tensor: shape=(4,), dtype=string, numpy=
array([b'h\xc3\x83llo', b'What is the weather tomorrow',
       b'G\xc3\xb6\xc3\xb6dnight', b'\xf0\x9f\x98\x8a'], dtype=object)>
```

对包含多个填充或稀疏格式字符串的张量，在调用 `tf.strings.unicode_encode` 前首先转换为 `tf.RaggedTensor`

- 稀疏张量，先转换为 `tf.RaggedTensor`

```python
tf.strings.unicode_encode(
    tf.RaggedTensor.from_sparse(batch_chars_sparse), output_encoding="UTF-8"
)
```

```txt
<tf.Tensor: shape=(4,), dtype=string, numpy=
array([b'h\xc3\x83llo', b'What is the weather tomorrow',
       b'G\xc3\xb6\xc3\xb6dnight', b'\xf0\x9f\x98\x8a'], dtype=object)>
```

- 填充张量，先转换为 `tf.RaggedTensor`

```python
tf.strings.unicode_encode(
    tf.RaggedTensor.from_tensor(batch_chars_padded, padding=-1), output_encoding="UTF-8"
)
```

```txt
<tf.Tensor: shape=(4,), dtype=string, numpy=
array([b'h\xc3\x83llo', b'What is the weather tomorrow',
       b'G\xc3\xb6\xc3\xb6dnight', b'\xf0\x9f\x98\x8a'], dtype=object)>
```

## 4. Unicode 操作

### 4.1 字符长度

通过 `tf.strings.length` 函数的 `unit` 参数指定如何计算字符长度。`unit` 默认为 `"BYTE"`，其它选项包括 `"UTF8_CHAR"` 或 `"UTF16_CHAR"`，用于指定编码字符串中 Unicode codepoints 数。

```python
# 在 UTF8 中最后一个字符占 4 个字节
thanks = u"Thanks 😊".encode("UTF-8")
num_bytes = tf.strings.length(thanks).numpy()
num_chars = tf.strings.length(thanks, unit="UTF8_CHAR").numpy()
print("{} bytes; {} UTF-8 characters".format(num_bytes, num_chars))
```

```txt
11 bytes; 8 UTF-8 characters
```

### 4.2 子字符串

`tf.strings.substr` 函数通过 `unit` 参数确定 `pos`  和 `len` 参数的 offsets。

```python
# 这里 unit='BYTE' (default). 返回长度为 1 的单个字节
tf.strings.substr(thanks, pos=7, len=1).numpy()
```

```txt
b'\xf0'
```

```python
# 设置 unit='UTF8_CHAR', 返回 1 个 4 字节字符
print(tf.strings.substr(thanks, pos=7, len=1, unit="UTF8_CHAR").numpy())
```

```txt
b'\xf0\x9f\x98\x8a'
```

### 4.3 拆分 Unicode 字符串

`tf.stringgs.unicode_split` 函数将 unicode 字符串拆分为子字符串或字符。

```python
tf.strings.unicode_split(thanks, "UTF-8").numpy()
```

```txt
array([b'T', b'h', b'a', b'n', b'k', b's', b' ', b'\xf0\x9f\x98\x8a'],
      dtype=object)
```

### 4.4 Byte offsets for characters

为了对齐 `tf.strings.unicode_decode` 生成的字符张量和原始字符串，知道每个字符开始位置的 offset 非常有用。`tf.strings.unicode_decode_with_offsets` 功能与 `unicode_decode` 类似，只是额外返回一个包含每个字符起始位置 offset 的张量。

```python
codepoints, offsets = tf.strings.unicode_decode_with_offsets(u"🎈🎉🎊", "UTF-8")

for (codepoint, offset) in zip(codepoints.numpy(), offsets.numpy()):
    print("At byte offset {}: codepoint {}".format(offset, codepoint))
```

```txt
At byte offset 0: codepoint 127880
At byte offset 4: codepoint 127881
At byte offset 8: codepoint 127882
```

## 5. Unicode scripts

每个 Unicode codepoint 属于一个称为 [script](https://en.wikipedia.org/wiki/Script_%28Unicode%29) codepoint 集合。字符的 script 有助于确定字符所属的语言。例如，已知 'Б' 在 Cyrillic script 中，表示包含该字符的文本可能来自 Slavic 语言，如俄罗斯或乌克兰语。

TF 提供 `tf.strings.unicode_script` 函数用于确定指定 codepoint 使用的 script。script 代码是 `int32` 值，与 [International Components for Unicode, ICU](http://site.icu-project.org/home) 的 [UScriptCode](icu-project.org/apiref/icu4c/uscript_8h.html) 值对应。

```python
uscript = tf.strings.unicode_script([33464, 1041])  # ['芸', 'Б']

print(uscript.numpy())  # [17, 8] == [USCRIPT_HAN, USCRIPT_CYRILLIC]
```

```txt
[17  8]
```

`tf.strings.unicode_script` 函数也可以用于包含 codepoints 的多维 `tf.Tensor` 或 `tf.RaggedTensor`：

```python
print(tf.strings.unicode_script(batch_chars_ragged))
```

```txt
<tf.RaggedTensor [[25, 25, 25, 25, 25],
 [25, 25, 25, 25, 0, 25, 25, 0, 25, 25, 25, 0, 25, 25, 25, 25, 25, 25, 25,
  0, 25, 25, 25, 25, 25, 25, 25, 25]                                      ,
 [25, 25, 25, 25, 25, 25, 25, 25, 25], [0]]>
```

## 6. 示例：简单分词

分词（segmentation）是将文本分割成类似单词单元的任务。对使用空格分割单词的文本，这很容易，但有些语言（如汉语、日语）不使用空格，有些语言（如德语）包含长的复合词，必须拆分才能理解其含义。在 web 文本中，不同语言和 script 经常混在一起，如 "NY株価" （纽约证券交易所）。

我们可以通过 script 的变化来近似单词边界，从而实现非常粗糙的分词（无需实现任何 ML 模型）。这适用于上面 "NY株価" 之类的字符串，也适用于使用空格分词的大多数语言，因为各种 script 均将空格字符作为 USCRIPT_COMMON，一种不同于实际文本的代码。

```python
# dtype: string; shape: [num_sentences]
#
# 要处理的句子。可以编辑此行尝试不同输入
sentence_texts = [u"Hello, world.", u"世界こんにちは"]
```

首先，将句子解码为字符代码点，并找到每个字符的 script：

```python
# dtype: int32; shape: [num_sentences, (num_chars_per_sentence)]
#
# sentence_char_codepoint[i, j] is the codepoint for the j'th character in
# the i'th sentence.
sentence_char_codepoint = tf.strings.unicode_decode(sentence_texts, "UTF-8")
print(sentence_char_codepoint)

# dtype: int32; shape: [num_sentences, (num_chars_per_sentence)]
#
# sentence_char_scripts[i, j] is the Unicode script of the j'th character in
# the i'th sentence.
sentence_char_script = tf.strings.unicode_script(sentence_char_codepoint)
print(sentence_char_script)
```

```txt
<tf.RaggedTensor [[72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 100, 46],
 [19990, 30028, 12371, 12435, 12395, 12385, 12399]]>
<tf.RaggedTensor [[25, 25, 25, 25, 25, 0, 0, 25, 25, 25, 25, 25, 0],
 [17, 17, 20, 20, 20, 20, 20]]>
```

使用 script 标识符确定单词边界。在每个句子的开头和 script 和前面字符不同的字符前面添加单词边界。

```python
# dtype: bool; shape: [num_sentences, (num_chars_per_sentence)]
#
# sentence_char_starts_word[i, j] is True if the j'th character in the i'th
# sentence is the start of a word.
sentence_char_starts_word = tf.concat(
    [
        tf.fill([sentence_char_script.nrows(), 1], True),
        tf.not_equal(sentence_char_script[:, 1:], sentence_char_script[:, :-1]),
    ],
    axis=1,
)

# dtype: int64; shape: [num_words]
#
# word_starts[i] is the index of the character that starts the i'th word (in
# the flattened list of characters from all sentences).
word_starts = tf.squeeze(tf.where(sentence_char_starts_word.values), axis=1)
print(word_starts)
```

```txt
tf.Tensor([ 0  5  7 12 13 15], shape=(6,), dtype=int64)
```

可以使用这些 start offsets 来构建包含所有 batches 的单词列表的 `RaggedTensor`。

```python
# dtype: int32; shape: [num_words, (num_chars_per_word)]
#
# word_char_codepoint[i, j] is the codepoint for the j'th character in the
# i'th word.
word_char_codepoint = tf.RaggedTensor.from_row_starts(
    values=sentence_char_codepoint.values, row_starts=word_starts
)
print(word_char_codepoint)
```

```txt
<tf.RaggedTensor [[72, 101, 108, 108, 111], [44, 32], [119, 111, 114, 108, 100], [46],
 [19990, 30028], [12371, 12435, 12395, 12385, 12399]]>
```

最后，将 `RaggedTensor` 包含的单词代码点转换为句子，并编码成 UTF-8 字符已便于阅读：

```python
# dtype: int64; shape: [num_sentences]
#
# sentence_num_words[i] is the number of words in the i'th sentence.
sentence_num_words = tf.reduce_sum(tf.cast(sentence_char_starts_word, tf.int64), axis=1)

# dtype: int32; shape: [num_sentences, (num_words_per_sentence), (num_chars_per_word)]
#
# sentence_word_char_codepoint[i, j, k] is the codepoint for the k'th character
# in the j'th word in the i'th sentence.
sentence_word_char_codepoint = tf.RaggedTensor.from_row_lengths(
    values=word_char_codepoint, row_lengths=sentence_num_words
)
print(sentence_word_char_codepoint)

tf.strings.unicode_encode(sentence_word_char_codepoint, "UTF-8").to_list()
```

```txt
<tf.RaggedTensor [[[72, 101, 108, 108, 111], [44, 32], [119, 111, 114, 108, 100], [46]],
 [[19990, 30028], [12371, 12435, 12395, 12385, 12399]]]>
[[b'Hello', b', ', b'world', b'.'],
 [b'\xe4\xb8\x96\xe7\x95\x8c',
  b'\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf']]
```

## 7. 参考

- https://www.tensorflow.org/text/guide/unicode
