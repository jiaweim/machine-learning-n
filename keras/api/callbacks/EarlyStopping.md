# EarlyStopping

Last updated: 2022-09-01, 17:06
@author Jiawei Mao
****

## 简介

```python
tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=0,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=False
)
```

当某个监控 metric 不再改善时停止训练。

假设训练的目标是最小化 loss，则监控的 metric 为 `'loss'`，模式为 `'min'`。`model.fit()` 训练循环在每个 epoch 结束时根据 `min_delta` 和 `patience` 检查 loss 是否不再减少。一旦发现 loss 不再减少，将 `model.stop_training` 设置为 True 并训练终止。

要监视的量值需要在 `logs` dict 中可用。为此，要在 `model.compile()` 中设置 loss 或 metrics。

## 参数

**monitor**

要监控的量值。

**min_delta**

监控量值变化小于 `min_delta` 被视为没有改进。

**patience**

如果在 `patience` 个 epochs 没有改进，停止训练。

**verbose**

详细模式：

- 0 表示静默
- 1 在 callback 执行时显示消息

**mode**

选择模式 {"auto", "min", "max"}：

- `min`，监控量值停止减小时终止训练
- `max`，监控量值停止增大时终止训练
- `auto`，根据监控量值名称确定方向

**baseline**

监控指标的基线值，如果模型相对基线没有改进，终止训练。

**restore_best_weights**

是否从监控指标最好的 epoch 恢复模型权重。False 表示使用训练最后一步获得的模型权重。True 则不管性能是否优于 `baseline`，都从最佳 epoch 恢复。如果没有任何 epoch 相对 `baseline` 改善，则训练持续 `patience` 个 epochs，然后从最佳 epoch 恢复权重。

## 示例

```python
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
# This callback will stop the training when there is no improvement in
# the loss for three consecutive epochs.
model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
model.compile(tf.keras.optimizers.SGD(), loss='mse')
history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
                    epochs=10, batch_size=1, callbacks=[callback],
                    verbose=0)
len(history.history['loss'])  # Only 4 epochs are run.
```

```txt
4
```

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
- https://keras.io/api/callbacks/early_stopping/
