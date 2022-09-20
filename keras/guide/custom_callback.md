# 自定义 callback

- [自定义 callback](#自定义-callback)
  - [简介](#简介)
  - [设置](#设置)
  - [Keras callbacks 概述](#keras-callbacks-概述)
  - [callback 方法概述](#callback-方法概述)
    - [Global 方法](#global-方法)
    - [Batch-level 方法](#batch-level-方法)
    - [Epoch-level 方法（仅训练）](#epoch-level-方法仅训练)
  - [一个简单示例](#一个简单示例)
    - [使用 logs dict](#使用-logs-dict)
  - [self.model 属性](#selfmodel-属性)
  - [Keras callback 应用示例](#keras-callback-应用示例)
    - [在 loss 最小时终止训练](#在-loss-最小时终止训练)
    - [学习率调整](#学习率调整)
  - [参考](#参考)

Last updated: 2022-09-20, 15:05
@author Jiawei Mao
****

## 简介

callback 是一个强大的工具，可以在训练、评估和推断期间自定义 Keras 模型的行为。例如，[tf.keras.callbacks.TensorBoard](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard) 可以使用 TensorBoard 可视化训练进度和结果，[tf.keras.callbacks.ModelCheckpoint](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint) 可以在训练时定期保存模型。

下面介绍介绍什么是 callback，它可以做什么，以及如何创建自己的 callback。

## 设置

```python
import tensorflow as tf
from tensorflow import keras
```

## Keras callbacks 概述

所有 callback 继承自 [keras.callbacks.Callback](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback) 类，并覆盖在训练、测试和推断等不同阶段自动调用的一组方法。

可以将 callback 列表（`callbacks` 关键字参数）传递给以下模型方法来使用：

- [keras.Model.fit()](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit)
- [keras.Model.evaluate()](https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate)
- [keras.Model.predict()](https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict)

## callback 方法概述

### Global 方法

- `on_(train|test|predict)_begin(self, logs=None)`

在 `fit`/`evaluate`/`predict` 前调用。

- `on_(train|test|predict)_end(self, logs=None)`

在 `fit`/`evaluate`/`predict` 结束时调用。

### Batch-level 方法

- `on_(train|test|predict)_batch_begin(self, batch, logs=None)`

在训练、测试、推理期间处理 batch 前调用。

- `on_(train|test|predict)_batch_end(self, batch, logs=None)`

在训练、测试、推理期间处理 batch 结束调用。`logs` 是包含 metrics 结果的 dict。

### Epoch-level 方法（仅训练）

- `on_epoch_begin(self, epoch, logs=None)`

在训练的 epoch 开始时调用。

- `on_epoch_end(self, epoch, logs=None)`

在训练的 epoch 结束时调用。

## 一个简单示例

下面通过一个具体的例子来展示 callback 的功能。首先定义一个 Keras Sequential 模型：

```python
def get_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(1, input_dim=784))
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.1),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )
    return model
```

然后，加载 MNIST 数据集：

```python
# 加载 MNIST 数据集，并预处理
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

# 只取前 1000 个样本
x_train = x_train[:1000]
y_train = y_train[:1000]
x_test = x_test[:1000]
y_test = y_test[:1000]
```

现在开始自定义一个 callback，在下面的各个阶段输出信息：

- `fit`/`evaluate`/`predict` 开始和结束时
- epoch 开始和结束
- 每个训练 batch 的开始和结束
- 每个评估 batch 的开始和结束
- 每个推断 batch 的开始和结束

```python
class CustomCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start testing; got log keys: {}".format(keys))

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop testing; got log keys: {}".format(keys))

    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start predicting; got log keys: {}".format(keys))

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop predicting; got log keys: {}".format(keys))

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))
```

试用这个 callback：

```python
model = get_model()
model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=1,
    verbose=0,
    validation_split=0.5,
    callbacks=[CustomCallback()],
)

res = model.evaluate(
    x_test, y_test, batch_size=128, verbose=0, callbacks=[CustomCallback()]
)

res = model.predict(x_test, batch_size=128, callbacks=[CustomCallback()])
```

```txt
Starting training; got log keys: []
Start epoch 0 of training; got log keys: []
...Training: start of batch 0; got log keys: []
...Training: end of batch 0; got log keys: ['loss', 'mean_absolute_error']
...Training: start of batch 1; got log keys: []
...Training: end of batch 1; got log keys: ['loss', 'mean_absolute_error']
...Training: start of batch 2; got log keys: []
...Training: end of batch 2; got log keys: ['loss', 'mean_absolute_error']
...Training: start of batch 3; got log keys: []
...Training: end of batch 3; got log keys: ['loss', 'mean_absolute_error']
Start testing; got log keys: []
...Evaluating: start of batch 0; got log keys: []
...Evaluating: end of batch 0; got log keys: ['loss', 'mean_absolute_error']
...Evaluating: start of batch 1; got log keys: []
...Evaluating: end of batch 1; got log keys: ['loss', 'mean_absolute_error']
...Evaluating: start of batch 2; got log keys: []
...Evaluating: end of batch 2; got log keys: ['loss', 'mean_absolute_error']
...Evaluating: start of batch 3; got log keys: []
...Evaluating: end of batch 3; got log keys: ['loss', 'mean_absolute_error']
Stop testing; got log keys: ['loss', 'mean_absolute_error']
End epoch 0 of training; got log keys: ['loss', 'mean_absolute_error', 'val_loss', 'val_mean_absolute_error']
Stop training; got log keys: ['loss', 'mean_absolute_error', 'val_loss', 'val_mean_absolute_error']
Start testing; got log keys: []
...Evaluating: start of batch 0; got log keys: []
...Evaluating: end of batch 0; got log keys: ['loss', 'mean_absolute_error']
...Evaluating: start of batch 1; got log keys: []
...Evaluating: end of batch 1; got log keys: ['loss', 'mean_absolute_error']
...Evaluating: start of batch 2; got log keys: []
...Evaluating: end of batch 2; got log keys: ['loss', 'mean_absolute_error']
...Evaluating: start of batch 3; got log keys: []
...Evaluating: end of batch 3; got log keys: ['loss', 'mean_absolute_error']
...Evaluating: start of batch 4; got log keys: []
...Evaluating: end of batch 4; got log keys: ['loss', 'mean_absolute_error']
...Evaluating: start of batch 5; got log keys: []
...Evaluating: end of batch 5; got log keys: ['loss', 'mean_absolute_error']
...Evaluating: start of batch 6; got log keys: []
...Evaluating: end of batch 6; got log keys: ['loss', 'mean_absolute_error']
...Evaluating: start of batch 7; got log keys: []
...Evaluating: end of batch 7; got log keys: ['loss', 'mean_absolute_error']
Stop testing; got log keys: ['loss', 'mean_absolute_error']
Start predicting; got log keys: []
...Predicting: start of batch 0; got log keys: []
...Predicting: end of batch 0; got log keys: ['outputs']
1/8 [==>...........................] - ETA: 0s...Predicting: start of batch 1; got log keys: []
...Predicting: end of batch 1; got log keys: ['outputs']
...Predicting: start of batch 2; got log keys: []
...Predicting: end of batch 2; got log keys: ['outputs']
...Predicting: start of batch 3; got log keys: []
...Predicting: end of batch 3; got log keys: ['outputs']
...Predicting: start of batch 4; got log keys: []
...Predicting: end of batch 4; got log keys: ['outputs']
...Predicting: start of batch 5; got log keys: []
...Predicting: end of batch 5; got log keys: ['outputs']
WARNING:tensorflow:Callback method `on_predict_batch_end` is slow compared to the batch time (batch time: 0.0004s vs `on_predict_batch_end` time: 0.0005s). Check your callbacks.
...Predicting: start of batch 6; got log keys: []
...Predicting: end of batch 6; got log keys: ['outputs']
...Predicting: start of batch 7; got log keys: []
...Predicting: end of batch 7; got log keys: ['outputs']
Stop predicting; got log keys: []
8/8 [==============================] - 0s 1ms/step
```

### 使用 logs dict

`logs` dict 包含 loss 值，以及 batch 或 epoch 结束时的所有 metrics。对上面的例子，则包含 loss 值和 mean_absolute_error。

```python
class LossAndErrorPrintingCallback(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        print(
            "Up to batch {}, the average loss is {:7.2f}.".format(batch, logs["loss"])
        )

    def on_test_batch_end(self, batch, logs=None):
        print(
            "Up to batch {}, the average loss is {:7.2f}.".format(batch, logs["loss"])
        )

    def on_epoch_end(self, epoch, logs=None):
        print(
            "The average loss for epoch {} is {:7.2f} "
            "and mean absolute error is {:7.2f}.".format(
                epoch, logs["loss"], logs["mean_absolute_error"]
            )
        )


model = get_model()
model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=2,
    verbose=0,
    callbacks=[LossAndErrorPrintingCallback()],
)

res = model.evaluate(
    x_test,
    y_test,
    batch_size=128,
    verbose=0,
    callbacks=[LossAndErrorPrintingCallback()],
)
```

```txt
Up to batch 0, the average loss is   31.56.
Up to batch 1, the average loss is  453.13.
Up to batch 2, the average loss is  309.70.
Up to batch 3, the average loss is  234.35.
Up to batch 4, the average loss is  189.27.
Up to batch 5, the average loss is  158.79.
Up to batch 6, the average loss is  137.01.
Up to batch 7, the average loss is  123.39.
The average loss for epoch 0 is  123.39 and mean absolute error is    6.04.
Up to batch 0, the average loss is    5.41.
Up to batch 1, the average loss is    5.40.
Up to batch 2, the average loss is    5.18.
Up to batch 3, the average loss is    5.04.
Up to batch 4, the average loss is    4.88.
Up to batch 5, the average loss is    4.73.
Up to batch 6, the average loss is    4.60.
Up to batch 7, the average loss is    4.75.
The average loss for epoch 1 is    4.75 and mean absolute error is    1.76.
Up to batch 0, the average loss is    5.27.
Up to batch 1, the average loss is    4.90.
Up to batch 2, the average loss is    4.91.
Up to batch 3, the average loss is    4.80.
Up to batch 4, the average loss is    4.90.
Up to batch 5, the average loss is    4.97.
Up to batch 6, the average loss is    4.90.
Up to batch 7, the average loss is    4.85.
```

## self.model 属性

`Callback` 的各个方法除了从 log 获得信息，还可以通过 `self.model` 访问当前模型。

在 callback 中使用 `self.model` 可以做很多事情，比如：

- 设置 `self.model.stop_training = True` 中断训练
- 修改 optimizer (`self.model.optimizer`) 的超参数，例如 `self.model.optimizer.learning_rate`
- 按周期保存模型
- 在每个 epoch 结尾记录 `model.predict` 在几个测试样本上的输出，以便在训练期间检查性能。
- 在每个 epoch 结尾提取中间特征的可视化，以监督模型随着时间的变化。

下面通过几个小例子看看。

## Keras callback 应用示例

### 在 loss 最小时终止训练

创建一个 `Callback`，通过设置 `self.model.stop_training` (boolean) 属性，实现在模型 loss 最小时终止训练。还提供一个 `patience` 参数，指定在达到局部最小值后，等待多少个 epoch 再终止训练。

对该功能 [tf.keras.callbacks.EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping) 提供了完整和通用实现。下面提供了一个简单实现：

```python
import numpy as np


class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
    """在 loss 达到最小时（即 loss 停止减少时）终止训练。

  Arguments:
      patience: 达到最小 loss 后等待的 epoch 数，在这些 epoch 后如果没有改进，终止训练。
  """
    def __init__(self, patience=0):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights 用来保存最佳权重
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # 训练在该 epoch 后终止
        self.stopped_epoch = 0
        # 初始化最佳 loss
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best): # 当前 loss 更小
            self.best = current
            self.wait = 0
            # 当前结果更好，所以更新最佳 weights
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


model = get_model()
model.fit(
    x_train,
    y_train,
    batch_size=64,
    steps_per_epoch=5,
    epochs=30,
    verbose=0,
    callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()],
)
```

```txt
Up to batch 0, the average loss is   26.43.
Up to batch 1, the average loss is  456.55.
Up to batch 2, the average loss is  310.41.
Up to batch 3, the average loss is  235.11.
Up to batch 4, the average loss is  189.38.
The average loss for epoch 0 is  189.38 and mean absolute error is    8.06.
Up to batch 0, the average loss is    6.38.
Up to batch 1, the average loss is    7.21.
Up to batch 2, the average loss is    6.38.
Up to batch 3, the average loss is    6.18.
Up to batch 4, the average loss is    5.99.
The average loss for epoch 1 is    5.99 and mean absolute error is    1.97.
Up to batch 0, the average loss is    5.09.
Up to batch 1, the average loss is    5.96.
Up to batch 2, the average loss is    5.45.
Up to batch 3, the average loss is    5.63.
Up to batch 4, the average loss is    6.12.
The average loss for epoch 2 is    6.12 and mean absolute error is    2.05.
Restoring model weights from the end of the best epoch.
Epoch 00003: early stopping

<keras.callbacks.History at 0x1a1ac435c70>
```

### 学习率调整

下面使用自定义 callback 在训练过程中动态调整学习率。

[tf.keras.callbacks.LearningRateScheduler](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler) 是一个更为通用的实现。

```python
class CustomLearningRateScheduler(keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

  Arguments:
      schedule: LR 调度函数，以 epoch 索引（0-based）和当前 LR 为输入，输出新的 LR
  """

    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # 从模型的 optimizer 查询当前 LR
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # 调用 schedule 函数获得计划 LR
        scheduled_lr = self.schedule(epoch, lr)
        # 在 epoch 开始前设置 optimizer 的 LR
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))


LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    (3, 0.05),
    (6, 0.01),
    (9, 0.005),
    (12, 0.001),
]


def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr


model = get_model()
model.fit(
    x_train,
    y_train,
    batch_size=64,
    steps_per_epoch=5,
    epochs=15,
    verbose=0,
    callbacks=[
        LossAndErrorPrintingCallback(),
        CustomLearningRateScheduler(lr_schedule),
    ],
)
```

```txt
Epoch 00000: Learning rate is 0.1000.
Up to batch 0, the average loss is   22.56.
Up to batch 1, the average loss is  461.56.
Up to batch 2, the average loss is  316.79.
Up to batch 3, the average loss is  240.73.
Up to batch 4, the average loss is  193.91.
The average loss for epoch 0 is  193.91 and mean absolute error is    8.27.

Epoch 00001: Learning rate is 0.1000.
Up to batch 0, the average loss is    7.08.
Up to batch 1, the average loss is    6.19.
Up to batch 2, the average loss is    5.66.
Up to batch 3, the average loss is    5.80.
Up to batch 4, the average loss is    5.68.
The average loss for epoch 1 is    5.68 and mean absolute error is    1.98.

Epoch 00002: Learning rate is 0.1000.
Up to batch 0, the average loss is    4.53.
Up to batch 1, the average loss is    4.83.
Up to batch 2, the average loss is    5.58.
Up to batch 3, the average loss is    5.21.
Up to batch 4, the average loss is    4.94.
The average loss for epoch 2 is    4.94 and mean absolute error is    1.74.

Epoch 00003: Learning rate is 0.0500.
Up to batch 0, the average loss is    4.56.
Up to batch 1, the average loss is    3.98.
Up to batch 2, the average loss is    3.46.
Up to batch 3, the average loss is    3.46.
Up to batch 4, the average loss is    3.60.
The average loss for epoch 3 is    3.60 and mean absolute error is    1.52.

Epoch 00004: Learning rate is 0.0500.
Up to batch 0, the average loss is    4.21.
Up to batch 1, the average loss is    3.87.
Up to batch 2, the average loss is    3.64.
Up to batch 3, the average loss is    3.88.
Up to batch 4, the average loss is    3.95.
The average loss for epoch 4 is    3.95 and mean absolute error is    1.59.

Epoch 00005: Learning rate is 0.0500.
Up to batch 0, the average loss is    3.51.
Up to batch 1, the average loss is    3.24.
Up to batch 2, the average loss is    3.24.
Up to batch 3, the average loss is    3.71.
Up to batch 4, the average loss is    3.92.
The average loss for epoch 5 is    3.92 and mean absolute error is    1.54.

Epoch 00006: Learning rate is 0.0100.
Up to batch 0, the average loss is    4.53.
Up to batch 1, the average loss is    3.79.
Up to batch 2, the average loss is    3.44.
Up to batch 3, the average loss is    3.76.
Up to batch 4, the average loss is    3.53.
The average loss for epoch 6 is    3.53 and mean absolute error is    1.43.

Epoch 00007: Learning rate is 0.0100.
Up to batch 0, the average loss is    2.59.
Up to batch 1, the average loss is    2.86.
Up to batch 2, the average loss is    3.10.
Up to batch 3, the average loss is    3.18.
Up to batch 4, the average loss is    3.02.
The average loss for epoch 7 is    3.02 and mean absolute error is    1.35.

Epoch 00008: Learning rate is 0.0100.
Up to batch 0, the average loss is    3.65.
Up to batch 1, the average loss is    3.06.
Up to batch 2, the average loss is    3.39.
Up to batch 3, the average loss is    3.73.
Up to batch 4, the average loss is    3.75.
The average loss for epoch 8 is    3.75 and mean absolute error is    1.52.

Epoch 00009: Learning rate is 0.0050.
Up to batch 0, the average loss is    3.25.
Up to batch 1, the average loss is    3.25.
Up to batch 2, the average loss is    3.08.
Up to batch 3, the average loss is    3.10.
Up to batch 4, the average loss is    3.13.
The average loss for epoch 9 is    3.13 and mean absolute error is    1.37.

Epoch 00010: Learning rate is 0.0050.
Up to batch 0, the average loss is    3.79.
Up to batch 1, the average loss is    3.78.
Up to batch 2, the average loss is    3.80.
Up to batch 3, the average loss is    4.04.
Up to batch 4, the average loss is    3.75.
The average loss for epoch 10 is    3.75 and mean absolute error is    1.49.

Epoch 00011: Learning rate is 0.0050.
Up to batch 0, the average loss is    3.70.
Up to batch 1, the average loss is    3.28.
Up to batch 2, the average loss is    3.46.
Up to batch 3, the average loss is    3.21.
Up to batch 4, the average loss is    3.10.
The average loss for epoch 11 is    3.10 and mean absolute error is    1.39.

Epoch 00012: Learning rate is 0.0010.
Up to batch 0, the average loss is    2.31.
Up to batch 1, the average loss is    2.39.
Up to batch 2, the average loss is    2.69.
Up to batch 3, the average loss is    2.76.
Up to batch 4, the average loss is    2.87.
The average loss for epoch 12 is    2.87 and mean absolute error is    1.32.

Epoch 00013: Learning rate is 0.0010.
Up to batch 0, the average loss is    2.27.
Up to batch 1, the average loss is    3.57.
Up to batch 2, the average loss is    3.77.
Up to batch 3, the average loss is    3.76.
Up to batch 4, the average loss is    3.64.
The average loss for epoch 13 is    3.64 and mean absolute error is    1.44.

Epoch 00014: Learning rate is 0.0010.
Up to batch 0, the average loss is    3.00.
Up to batch 1, the average loss is    3.14.
Up to batch 2, the average loss is    3.01.
Up to batch 3, the average loss is    3.16.
Up to batch 4, the average loss is    3.25.
The average loss for epoch 14 is    3.25 and mean absolute error is    1.41.

<keras.callbacks.History at 0x1a1ad1f9ee0>
```

## 参考

- https://www.tensorflow.org/guide/keras/custom_callback
- https://keras.io/guides/writing_your_own_callbacks/
