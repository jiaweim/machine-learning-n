# tf.strings.reduce_join

2022-03-02, 23:21
****

## 简介

```python
tf.strings.reduce_join(
    inputs, axis=None, keepdims=False, separator='', name=None
)
```

将所有字符串连接成一个字符串，是逐元素操作 [tf.strings.join](join.md) 的 reduce 操作。

```python
>>> tf.strings.reduce_join([['abc','123'],
...                         ['def','456']]).numpy()
b'abc123def456'
>>> tf.strings.reduce_join([['abc','123'],
...                         ['def','456']], axis=-1).numpy()
array([b'abc123', b'def456'], dtype=object)
>>> tf.strings.reduce_join([['abc','123'],
...                         ['def','456']],
...                        axis=-1,
...                        separator=" ").numpy()
array([b'abc 123', b'def 456'], dtype=object)
```

## 参数

|参数|说明|
|---|---|
|inputs|A tf.string tensor.|
|axis|Which axis to join along. The default behavior is to join all elements, producing a scalar.|
|keepdims|If true, retains reduced dimensions with length 1.|
|separator|a string added between each string being joined.|
|name|A name for the operation (optional).|

## 参考

- https://www.tensorflow.org/api_docs/python/tf/strings/reduce_join
