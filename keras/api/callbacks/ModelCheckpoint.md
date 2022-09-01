# ModelCheckpoint

Last updated: 2022-09-01, 17:01
****

## 简介

```python
tf.keras.callbacks.ModelCheckpoint(
    filepath,
    monitor='val_loss',
    verbose=0,
    save_best_only=False,
    save_weights_only=False,
    mode='auto',
    save_freq='epoch',
    options=None,
    initial_value_threshold=None,
    **kwargs
)
```

以指定频率保存 Keras 模型或权重的 callback。

`ModelCheckpoint` callback 通过 `model.fit()` 与训练结合使用，以指定时间间隔保存模型或权重为 checkpoint 文件，以便稍后可以加载模型或权重，从而从保存的状态继续训练。

该 callback 提供了如下选项：

- 是只保留到目前为止**性能最佳**的模型，还是不管性能，每个 epoch 结束时都保存模型；
- **最佳的定义**：要监控的指标，以及应该最大化还是最小化；
- 保存的频率，目前支持在每个 epoch 结束时保存，或指定训练 batches 后保存；
- 是只保存权重，还是保存整个模型。

> **[!NOTE]**
> 如果出现 `WARNING:tensorflow:Can save best model only with <name> available, skipping` 信息，可以参考 `monitor` 参数说明。

## 参数

**filepath**

保存模型文件的路径，string 或 `PathLike`，例如 `filepath = os.path.join(working_dir, 'ckpt', file_name)`。`filepath` 可以包含命名格式化选项。例如，如果 `filepath` 为 `weights.{epoch:02d}-{val_loss:.2f}.hdf5`，则 model checkpoint 文件名包含 epoch 号和 validation loss。`filepath` 目录不应该被其它 callback 使用，以避免冲突。

**monitor**

要监控指标的名称。指标一般通过 `Model.compile` 方法设置。注意：

- 在名称前加 "val_" 前缀以监控 validation 指标；
- 使用 "loss" 或 "val_loss" 以监控模型的总损失；
- 如果用字符串指定指标，如 "accuracy"，传入相同的字符串（带或不带 "val_" 前缀）；
- 如果传入 `metrics.Metric` 对象，`monitor` 应该设置为 `metric.name`；
- 如果不确定指标名称，可以检查 `history=model.fit()` 返回的 `history.history` dict；
- 多输出模型的指标名称包含额外的前缀。

**verbose**

详细模式，0 或 1:

- 0 silent
- 1 在 callback 执行时显示消息

**save_best_only**

`save_best_only=True` 只在模型被认为时目前最好时保存。如果 `filepath` 不包含格式化选项，例如 `{epoch}`，则每个新的更好的模型将覆盖之前保存的模式。

**mode**

{'auto', 'min', 'max'} 之一。

如果 `save_best_only=True`，则根据监视指标的最大化或最小化来决定是否覆盖保存文件。对 `val_acc` 应为 `max`，对 `val_loss` 应为 `min`。在 `auto` 模式，如果监控的指标为 `acc` 或以 'fmeasure' 开头，则模式为 `max`，对余下的则为 `min`。

**save_weights_only**

True 表示只保存模型的权重 `model.save_weights(filepath)`，否则保存整个模型 `model.save(filepath)`。

**save_freq**

'epoch' 或 integer。当使用 `'epoch'` 时，callback 在每个 epoch 后保存模型。当使用 integer，则在这些 batch 后保存模型。如果 `Model` 使用 `steps_per_execution=N` 选项进行编译，则每 Nth batch 检查保存条件。注意，如果保存和 epoch 没对齐，则监控指标可能不可靠（它可能只反应一个 batch，因为指标在每个 epoch 结束会重置）。默认 'epoch'。

**options**

`save_weights_only` 为 True 时可选的 `tf.train.CheckpointOptions` 对象 或 `save_weights_only` 为 False 时可选的 `tf.saved_model.SaveOptions` 对象。

**initial_value_threshold**

指标的最佳值（浮点数）。`save_best_value=True` 时适用。当模型的性能优于该值时才保存模型权重。

## 示例

```python
model.compile(loss=..., optimizer=..., metrics=["accuracy"])

EPOCHS = 10
checkpoint_filepath = "/tmp/checkpoint"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
)

# 如果模型是目前为止最好的模型，则在每个 epoch 后保存模型权重
model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])

# The model weights (that are considered the best) are loaded into the model.
model.load_weights(checkpoint_filepath)
```

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
- https://keras.io/api/callbacks/model_checkpoint/
