# pad_sequences

- [pad\_sequences](#pad_sequences)
  - [简介](#简介)
  - [示例](#示例)
  - [参考](#参考)

2022-06-07, 17:18
****

## 简介

将序列填充到长度相同。

```py
tf.keras.utils.pad_sequences(
    sequences,
    maxlen=None,
    dtype='int32',
    padding='pre',
    truncating='pre',
    value=0.0
)
```

该函数将一组序列（`num_samples` 条）转换为 `(num_samples, num_timesteps)` 的 2D Numpy 数组。`num_timesteps` 要么使用提供的 `maxlen` 参数，要么是最长序列的长度。

- 长度小于 `num_timesteps` 的序列用 `value` 填充到 `num_timesteps` 长度。
- 长度大于 `num_timesteps` 的序列截断到该长度。

填充或截断的位置由参数 `padding` 和 `truncating` 决定。**默认在序列开头填充或截断**。

**参数：**

- **sequences**

序列 list，每条序列是 integer list。

- **maxlen**：Optional Int

最大长度。如果不提供，则为 `sequences` 中最长序列的长度。

- **dtype**	(Optional, defaults to "int32")

输出序列的类型。对变长字符串序列，可以用 `object`。

- **padding** - String, "pre" or "post" (optional, defaults to `"pre"`)

在每条序列的前面还是后面填充。

- **truncating** - String, "pre" or "post" (optional, defaults to **"pre"**)

对长度超过 `maxlen` 的序列要截断，截取位置可是开头或末尾。

- **value**	Float or String, 填充值 (Optional, defaults to `0.`)

**返回：**

shape 为 `(len(sequences), maxlen)` 的 numpy 数组。

## 示例

- 默认在开头填充 0

```py
>>> import tensorflow as tf
>>> sequence = [[1], [2, 3], [4, 5, 6]]
>>> tf.keras.preprocessing.sequence.pad_sequences(sequence)
array([[0, 0, 1],
       [0, 2, 3],
       [4, 5, 6]])
```

- 设置填充 -1

```py
>>> tf.keras.preprocessing.sequence.pad_sequences(sequence, value=-1)
array([[-1, -1,  1],
       [-1,  2,  3],
       [ 4,  5,  6]])
```

- 设置在末尾填充

```py
>>> tf.keras.preprocessing.sequence.pad_sequences(sequence, padding='post')
array([[1, 0, 0],
       [2, 3, 0],
       [4, 5, 6]])
```

- 设置最大长度

```py
>>> tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=2)
array([[0, 1],
       [2, 3],
       [5, 6]])
```

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/utils/pad_sequences
