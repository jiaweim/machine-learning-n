# RepeatVector

Last updated: 2022-08-01, 15:04
@author Jiawei Mao
***

## 简介

```python
tf.keras.layers.RepeatVector(
    n, **kwargs
)
```

将输入重复 n 次。

例如：

```python
model = Sequential()
model.add(Dense(32, input_dim=32))
# now: model.output_shape == (None, 32)
# note: `None` is the batch dimension

model.add(RepeatVector(3))
# now: model.output_shape == (None, 3, 32)
```

## 参数

- n

整数，重复次数。

输入 shape: 2D 张量 `(num_samples, features)`

输出 shape：3D 张量 `(num_samples, n, features)`

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/layers/RepeatVector
