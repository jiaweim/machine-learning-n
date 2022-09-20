# Callbacks

***

## 简介

callback 是一个可以在训练的各个阶段执行操作的对象，例如在 epoch 的开始或结束，在单个 batch 的开头或结尾等。

callback 的功能：

- 在训练的每个 batch 后输出 TensorBoard 日志，以监控模型指标
- 定期将模型保存到硬盘
- 提前停止训练
- 训练期间查看模型的内部状态和统计信息
- ......

## fit() 中使用 callbacks

将 callback 列表传递给 `fit()` 的 `callbacks` 参数：

```python
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]
model.fit(dataset, epochs=10, callbacks=my_callbacks)
```

## 参考

- https://keras.io/api/callbacks/
- https://www.tensorflow.org/api_docs/python/tf/keras/callbacks
