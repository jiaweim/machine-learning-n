# preprocessing.sequence

- [preprocessing.sequence](#preprocessingsequence)
  - [pad_sequences](#pad_sequences)

## pad_sequences

将序列填充到相同的长度。

```python
tf.keras.preprocessing.sequence.pad_sequences(
    sequences, maxlen=None, dtype='int32', padding='pre',
    truncating='pre', value=0.0
)
```

该函数将序列（整数列表）列表（长度为 `num_samples`）转换为 2D numpy 数组，shape 为 `(num_samples, num_timesteps)`。

其中 `num_timesteps`：

- 如果提供了 `maxlen`，则采用该值；
- 否则为最长序列长度。

长度小于 `num_timesteps` 的序列，以 `value` 填充到该长度。

长度大于 `num_timesteps` 的序列，则截断到该长度。

填充或截断的位置由参数 `padding` 和 `truncating` 设置。默认对序列开头填充或截断。例如：

```python
>>> import tensorflow as tf
>>> sequence = [[1], [2, 3], [4, 5, 6]]
>>> tf.keras.preprocessing.sequence.pad_sequences(sequence)
array([[0, 0, 1],
       [0, 2, 3],
       [4, 5, 6]])
```
