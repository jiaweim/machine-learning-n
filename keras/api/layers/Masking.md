# Masking

- [Masking](#masking)
  - [简介](#简介)
  - [示例](#示例)
  - [参考](#参考)

2022-06-07, 16:58
****

## 简介

根据 mask 值跳过特定时间步。

```py
tf.keras.layers.Masking(
    mask_value=0.0, **kwargs
)
```

对输入张量的每个时间步，如果该时间步的所有值都等于 `mask_value`，则在所有下游 layer 中将跳过该时间步。

如果下游 layer 中有不支持 mask 的，抛出异常。

## 示例

将 shape 为 `(samples, timesteps, features)` 的 numpy 数组 `x` 传入 `LSTM` 层。希望跳过 #3 和 #5 这两个时间步，因为缺少对应的数据，可以这么干：

- 设置 `x[:, 3, :] = 0.` 和 `x[:, 5, :] = 0.`
- 在 LSTM 层前面插入 `Masking` 层，设置 `mask_value=0.`

```py
samples, timesteps, features = 32, 10, 8
inputs = np.random.random([samples, timesteps, features]).astype(np.float32)
inputs[:, 3, :] = 0.
inputs[:, 5, :] = 0.

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Masking(mask_value=0.,
                                  input_shape=(timesteps, features)))
model.add(tf.keras.layers.LSTM(32))

output = model(inputs)
# 在 LSTM 计算中会跳过 3 和 5 两个时间步
```

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/layers/Masking
