# tf.keras.utils.to_categorical

## 简介

```python
tf.keras.utils.to_categorical(
    y, num_classes=None, dtype='float32'
)
```

将类向量（整数）转换为 binary class matrix。用于 onehot 编码。

**参数：**

- **y**

类数组对象，将其类别（class）转换为矩阵（0 到 `num_classes - 1` 的整数）

- **num_classes**

总类别数，`None` 则使用 `max(y) + 1`。


dtype	The data type expected by the input. Default: 'float32'.
