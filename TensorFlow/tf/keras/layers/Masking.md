# Masking

- [Masking](#masking)
  - [简介](#简介)
  - [示例](#示例)
  - [个人总结](#个人总结)
  - [参考](#参考)

Last updated: 2023-01-18, 10:18
****

## 简介

根据 mask 值跳过特定时间步。

```python
tf.keras.layers.Masking(
    mask_value=0.0, **kwargs
)
```

对输入张量的每个时间步（张量的 #1 维度），如果输入张量在该时间步的所有值都等于 `mask_value`，则下游 layer 将跳过该时间步。

如果下游 layer 不支持 mask，抛出异常。

## 示例

将 shape 为 `(samples, timesteps, features)` 的 numpy 数组 `x` 传入 `LSTM` 层。因为缺少对应的数据希望跳过 #3 和 #5 这两个时间步，操作方式：

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
# 在 LSTM 计算中会跳过 batch 所有样本的 3 和 5 两个时间步
```

## 个人总结

-  `Masking` 需要输入为 3D，即 (batch, sequence, features)
-  `Masking` 不修改输入，而是额外添加了一个 `_keras_mask` 属性
-  使用 `compute_mask()` 更新 mask，默认直接返回上一个 mask
-  可以用 `supports_masking` 属性查看 `Layer` 是否支持 mask

支持 masking 的 keras layers：`SimpleRNN`, `GRU`, `LSTM`, `Bidirectional`, `Dense`, `TimeDistributed`, `Add` 等。

如果 mask 从输入传播到输出，那么也应用于 loss 函数，即 mask 时间步不影响 loss。

`Masking` 等价于 `tf.math.reduce_any(tf.math.not_equal(X, 0), axis=-1)`，即将最后一个维度全为 0 的时间步设为 False。

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/layers/Masking
