# tf.strings.join

2022-03-02, 23:24
****

## 简介

```python
tf.strings.join(
    inputs, separator='', name=None
)
```

对字符串张量列表进行逐元素串联。

对相同 shape 的字符串张量列表，对所有张量中相同索引的字符串进行串联。

```python
>>> tf.strings.join(['abc','def']).numpy()
b'abcdef'
>>> tf.strings.join([['abc','123'],
...                  ['def','456'],
...                  ['ghi','789']]).numpy()
array([b'abcdefghi', b'123456789'], dtype=object)
>>> tf.strings.join([['abc','123'],
...                  ['def','456']],
...                  separator=" ").numpy()
array([b'abc def', b'123 456'], dtype=object)
```

## 参数

|参数|说明|
|---|---|
|inputs|A list of `tf.Tensor` objects of same size and `tf.string` dtype.|
|separator|A string added between each string being joined.|
|name|A name for the operation (optional).|

## 参考

- https://www.tensorflow.org/api_docs/python/tf/strings/join
