# 使用内置方法训练和评估

- [使用内置方法训练和评估](#使用内置方法训练和评估)
  - [简介](#简介)
  - [API 概述](#api-概述)
  - [compile()：指定 loss, metrics 和 optimizr](#compile指定-loss-metrics-和-optimizr)
    - [内置 optimizer, loss 和 metric](#内置-optimizer-loss-和-metric)
    - [自定义 loss](#自定义-loss)
    - [自定义 metric](#自定义-metric)
    - [不符合标准签名的 loss 和 metric](#不符合标准签名的-loss-和-metric)
    - [自动设置验证 holdout](#自动设置验证-holdout)
  - [使用 tf.data 数据集进行训练和验证](#使用-tfdata-数据集进行训练和验证)
    - [使用验证集](#使用验证集)
  - [其它输入格式](#其它输入格式)
  - [keras.utils.Sequence 输入](#kerasutilssequence-输入)
  - [样本加权和类别加权](#样本加权和类别加权)
    - [类别加权](#类别加权)
    - [样本加权](#样本加权)
  - [多输入/多输出模型的数据](#多输入多输出模型的数据)
  - [callback](#callback)
    - [内置 callback](#内置-callback)
    - [自定义 callback](#自定义-callback)
  - [Checkpoint 模型](#checkpoint-模型)
  - [学习率设置](#学习率设置)
    - [在 optimizer 设置预定义 LR](#在-optimizer-设置预定义-lr)
    - [使用 callback 实现动态 LR schedule](#使用-callback-实现动态-lr-schedule)
  - [训练期间可视化 loss 和 metrics](#训练期间可视化-loss-和-metrics)
    - [使用 TensorBoard callback](#使用-tensorboard-callback)
  - [参考](#参考)

Last updated: 2022-09-27, 15:45
****

## 简介

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

下面介绍如何使用内置 API 进行训练（`Model.fit()`）、评估（`Model.evaluate()`）和推理（`Model.predict()`）。

如果想自定义 `fit()` 中训练循环步骤，请参考 [Customize what happens in Model.fit](https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit/)。

如果想从头开始编写自己的训练和评估循环，可参考 [Writing a training loop from scratch](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch/)。

通常，无论是使用内置循环还是自己编写，在各种 Keras 模型中（Sequential 模型、函数 API 模型，子类模型），模型的训练和评估都以相同的方式工作。

## API 概述

TF 支持的数据形式包括 NumPy 数组（可以全部载入内存的小数据集）和 `tf.data.Dataset` 对象。接下来，使用 NumPy 数组类型的 MNIST 数据集，演示如何使用 optimizers、losses 和 metrics。

考虑如下模型（这里用的Functional API，使用 Sequential 模型或子类模型亦可）：

```python
inputs = keras.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(10, activation="softmax", name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```

下面是典型的端到端工作流：

- 训练
- 在验证集（holdout）上进行验证
- 在测试集上评估

本例使用 MNIST 数据集进行演示：

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 预处理数据: 转换为 1 维，类型转换为 float32，特征值缩放到 [0, 1]
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

# 从训练集中分出 10000 个样本用作验证
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]
```

指定训练配置（optimizer, loss, metrics）：

```python
model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer
    # loss 函数
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # 记录的 metric 列表
    metrics=[keras.metrics.SparseCategoricalAccuracy(), ]
)
```

调用 `fit()` 训练模型，该方法将整个数据集拆分成大小为 `batch_size` 的多个批次，并迭代整个数据集 `epochs` 次：

```python
print("Fit model on training data")
history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=2,
    # 验证集，每个 epoch 后查看模型在验证集上的 loss 和 metrics
    validation_data=(x_val, y_val)
)
```

```txt
Fit model on training data
Epoch 1/2
782/782 [==============================] - 6s 6ms/step - loss: 0.3430 - sparse_categorical_accuracy: 0.9034 - val_loss: 0.2571 - val_sparse_categorical_accuracy: 0.9186
Epoch 2/2
782/782 [==============================] - 4s 5ms/step - loss: 0.1590 - sparse_categorical_accuracy: 0.9523 - val_loss: 0.1434 - val_sparse_categorical_accuracy: 0.9563
```

返回的 `history` 对象持有训练期间的 loss 和 metric 值：

```python
history.history
```

```txt
{'loss': [0.3430411219596863, 0.1589580923318863],
 'sparse_categorical_accuracy': [0.9034000039100647, 0.9523400068283081],
 'val_loss': [0.2571413516998291, 0.14339785277843475],
 'val_sparse_categorical_accuracy': [0.9186000227928162, 0.9563000202178955]}
```

> 因为训练了 2 个 epochs，所以 loss 和 metric 列表值的长度都是 2.

使用 `evaluate()` 在测试集上评估模型：

```python
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)

# 使用 `predict` 预测新数据
print("Generate predictions for 3 samples")
predictions = model.predict(x_test[:3])
print("predictions shape:", predictions.shape)
```

```txt
Evaluate on test data
79/79 [==============================] - 0s 4ms/step - loss: 0.1423 - sparse_categorical_accuracy: 0.9585
test loss, test acc: [0.14227981865406036, 0.9585000276565552]
Generate predictions for 3 samples
1/1 [==============================] - 0s 60ms/step
predictions shape: (3, 10)
```

下面对该工作流的每个部分进行详细讨论。

## compile()：指定 loss, metrics 和 optimizr

使用 `fit()` 训练模型，需要指定 loss 函数，optimizer，还可以指定一些要记录的 metrics。

这些信息可以作为参数传递给 `compile()` 方法：

```python
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
```

`metrics` 参数为 list 类型，模型可以包含任意个 metrics。

如果模型有多个输出，可以为每个输出指定不同的 loss 和 metric，还可以调节各个输出对模型总 loss 的贡献。后面会详细介绍。

如果不需要额外配置，即接受 optimizer, loss 和 metric 的默认值，则可以使用字符串名称指定它们：

```python
model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)
```

为了便于复用，下面将模型的定义和编译放在函数中，后面我们会在不同示例中多次调用这些函数：

```python
def get_uncompiled_model():
    inputs = keras.Input(shape=(784,), name="digits")
    x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_compiled_model():
    model = get_uncompiled_model()
    model.compile(
        optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model
```

### 内置 optimizer, loss 和 metric

通常不需要从头创建自己的 loss, metric 或 optimizer，因为你需要的 Keras API 可能都有。

Optimizers：

- `SGD()` (可选动量)
- `RMSprop()`
- `Adam()`
- ...

Losses：

- `MeanSquaredError()`
- `KLDivergence()`
- `CosineSimilarity()`
- ...

Metrics：

- `AUC()`
- `Precision()`
- `Recall()`
- ...

### 自定义 loss

Keras 提供了两种自定义 loss 的方法。

**第一种**，创建接受输入 `y_true` 和 `y_pred` 的函数。例如，将真实值和预测值之间的均方差作为 loss：

```python
def custom_mean_squared_error(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))


model = get_uncompiled_model()
model.compile(optimizer=keras.optimizers.Adam(), loss=custom_mean_squared_error)

# 为了使用 MSE，对标签进行 one-hot 编码
y_train_one_hot = tf.one_hot(y_train, depth=10)
model.fit(x_train, y_train_one_hot, batch_size=64, epochs=1)
```

```txt
782/782 [==============================] - 3s 3ms/step - loss: 0.0159
<keras.callbacks.History at 0x1b32735a9a0>
```

**第二种**方法，如果所需 loss 除了 `y_true` 和 `y_pred`，还需要其它参数，可以扩展 `tf.keras.losses.Loss` 类，并实现以下两个方法：

- `__init__(self)`：在调用损失函数时可以接受参数。
- `call(self, y_true, y_pred)`：使用目标值 `y_true` 和模型预测值 `y_pred` 计算模型损失。

假设你要使用均方差，但添加额外项，使预测值远离 0.5（假设目标分类值为 one-hot 编码，值在 0 到 1 之间）。

创建方法：

```python
class CustomMSE(keras.losses.Loss):
    def __init__(self, regularization_factor=0.1, name="custom_mse"):
        super().__init__(name=name)
        self.regularization_factor = regularization_factor

    def call(self, y_true, y_pred):
        mse = tf.math.reduce_mean(tf.square(y_true - y_pred))
        reg = tf.math.reduce_mean(tf.square(0.5 - y_pred))
        return mse + reg * self.regularization_factor


model = get_uncompiled_model()
model.compile(optimizer=keras.optimizers.Adam(), loss=CustomMSE())

y_train_one_hot = tf.one_hot(y_train, depth=10)
model.fit(x_train, y_train_one_hot, batch_size=64, epochs=1)
```

```txt
782/782 [==============================] - 3s 3ms/step - loss: 0.0390
<keras.callbacks.History at 0x1b328948f70>
```

### 自定义 metric

如果 Keras API 不包含你所需的 metric，则可以通过扩展 `tf.keras.metrics.Metric` 类来自定义。需要实现 4 个方法：

- `__init__(self)`，创建实现 metric 所需的状态变量
- `update_state(self, y_true, y_pred, sample_weight=None)`，使用 `y_true` 和 `y_pred` 更新状态变量
- `result(self)` 使用状态变量计算最终结果
- `reset_state(self)` 重新初始化 metric 的状态变量

状态更新和结果计算分开放在 `update_state()` 和 `result()` 中，因为有时候计算结果可能很耗时贵，只能周期性进行。

下面演示如何实现 `CategoricalTruePositives` 指标，该指标计算分类正确的样本数目：

```python
class CategoricalTruePositives(keras.metrics.Metric):
    def __init__(self, name="categorical_true_positives", **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        values = tf.cast(y_true, "int32") == tf.cast(y_pred, "int32")
        values = tf.cast(values, "float32")
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_state(self):
        # 在每个 epoch 开始指标的状态被重置
        self.true_positives.assign(0.0)


model = get_uncompiled_model()
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[CategoricalTruePositives()],
)
model.fit(x_train, y_train, batch_size=64, epochs=3)
```

```txt
Epoch 1/3
782/782 [==============================] - 5s 6ms/step - loss: 0.3395 - categorical_true_positives: 45154.0000
Epoch 2/3
782/782 [==============================] - 4s 6ms/step - loss: 0.1592 - categorical_true_positives: 47583.0000
Epoch 3/3
782/782 [==============================] - 5s 6ms/step - loss: 0.1180 - categorical_true_positives: 48225.0000
<keras.callbacks.History at 0x1b32e0fcf40>
```

### 不符合标准签名的 loss 和 metric

绝大多数的 loss 和 metric 可以从 `y_true` 和 `y_pred` 计算，其中 `y_pred` 是模型预测结果，但并非全部如此。例如，正则化损失可能只需要激活值，并且该激活值不一定是模型的输出。

此时可以在自定义 layer 的 `call` 方法中调用 `self.add_loss(loss_value)`。以这种方式添加的 loss 在训练过程中被添加到 "main" loss 中（通过 `compile()` 设置）。例如，添加激活正则化（Keras 内置有激活正则化，此处为了演示提供了一个实现）:

```python
class ActivityRegularizationLayer(layers.Layer):
    def call(self, inputs):
        self.add_loss(tf.reduce_sum(inputs) * 0.1)
        return inputs  # Pass-through layer.


inputs = keras.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)

# 插入激活正则化 layer
x = ActivityRegularizationLayer()(x)

x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(10, name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

# 由于正则化部分，使得损失值比以前大很多
model.fit(x_train, y_train, batch_size=64, epochs=1)
```

```txt
782/782 [==============================] - 4s 5ms/step - loss: 2.4859
<keras.callbacks.History at 0x1b460fc5640>
```

可以使用 `add_metric()` 对 logging metric 做相同操作：

```python
class MetricLoggingLayer(layers.Layer):
    def call(self, inputs):
        # The `aggregation` argument defines
        # how to aggregate the per-batch values
        # over each epoch:
        # in this case we simply average them.
        self.add_metric(
            keras.backend.std(inputs), name="std_of_activation", aggregation="mean"
        )
        return inputs  # Pass-through layer.


inputs = keras.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)

# Insert std logging as a layer.
x = MetricLoggingLayer()(x)

x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(10, name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)
model.fit(x_train, y_train, batch_size=64, epochs=1)
```

```txt
782/782 [==============================] - 4s 5ms/step - loss: 0.3426 - std_of_activation: 0.9398
<keras.callbacks.History at 0x2c466a74be0>
```

在 Keras 函数 API 中，也可以调用 `model.add_loss(loss_tensor)` 或 `model.add_metric(metric_tensor, name, aggregation)`。例如：

```python
inputs = keras.Input(shape=(784,), name="digits")
x1 = layers.Dense(64, activation="relu", name="dense_1")(inputs)
x2 = layers.Dense(64, activation="relu", name="dense_2")(x1)
outputs = layers.Dense(10, name="predictions")(x2)
model = keras.Model(inputs=inputs, outputs=outputs)

model.add_loss(tf.reduce_sum(x1) * 0.1)

model.add_metric(keras.backend.std(x1), name="std_of_activation", aggregation="mean")

model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)
model.fit(x_train, y_train, batch_size=64, epochs=1)
```

```txt
782/782 [==============================] - 5s 5ms/step - loss: 2.4936 - std_of_activation: 0.0019
<keras.callbacks.History at 0x2c46796c490>
```

注意，当通过 `add_loss()` 传递 loss 时，调用 `compile()` 就可以不设置 loss 函数，因为模型已有需最小化的 loss。

考虑下面的 `LogisticEndpoint` layer：它以 targets 和 logits 为输入，通过 `add_loss()` 记录交叉熵损失。还通过 `add_metric()` 记录分类精度。

```python
class LogisticEndpoint(keras.layers.Layer):
    def __init__(self, name=None):
        super(LogisticEndpoint, self).__init__(name=name)
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        self.accuracy_fn = keras.metrics.BinaryAccuracy()

    def call(self, targets, logits, sample_weights=None):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        loss = self.loss_fn(targets, logits, sample_weights)
        self.add_loss(loss)

        # Log accuracy as a metric and add it
        # to the layer using `self.add_metric()`.
        acc = self.accuracy_fn(targets, logits, sample_weights)
        self.add_metric(acc, name="accuracy")

        # Return the inference-time prediction tensor (for `.predict()`).
        return tf.nn.softmax(logits)
```

也可以在包含两个输入（输入数据和 targets）的模型中使用，不使用 `loss` 参数进行编译，如下：

```python
import numpy as np

inputs = keras.Input(shape=(3,), name="inputs")
targets = keras.Input(shape=(10,), name="targets")
logits = keras.layers.Dense(10)(inputs)
predictions = LogisticEndpoint(name="predictions")(logits, targets)

model = keras.Model(inputs=[inputs, targets], outputs=predictions)
model.compile(optimizer="adam")  # No loss argument!

data = {
    "inputs": np.random.random((3, 3)),
    "targets": np.random.random((3, 10)),
}
model.fit(data)
```

```txt
1/1 [==============================] - 0s 363ms/step - loss: 1.0555 - binary_accuracy: 0.0000e+00
<keras.callbacks.History at 0x2c44a1ffca0>
```

下面会详细介绍如何训练多输入模型。

### 自动设置验证 holdout

在上面第一个端到端示例中，使用 `validation_data` 参数将 numpy 数组 `(x_val, y_val)` 传递给模型，以评估模型在每个 epoch 结束后的 validation loss 和 validation metric。

这里还有另一个选择：参数 `validation_split` 自动保留训练数据的一部分用作验证。该参数表示留作验证数据的比例，所以在 0 到 1 之间。例如，`validation_split=0.2` 表示使用 20% 的数据进行验证，`validation_split=0.6` 表示使用 60% 的数据进行验证。

验证数据的采样方式：在洗牌（shuffle）前，对 `fit()` 接受到的数组的最后 x% 样本用作验证。

不过要注意，只有在使用 NumPy 数组类型数据进行训练时才能用 `validation_split`。

```python
model = get_compiled_model()
model.fit(x_train, y_train, batch_size=64, validation_split=0.2, epochs=1)
```

```txt
625/625 [==============================] - 4s 6ms/step - loss: 0.3724 - sparse_categorical_accuracy: 0.8942 - val_loss: 0.2298 - val_sparse_categorical_accuracy: 0.9294
<keras.callbacks.History at 0x2c467a279a0>
```

## 使用 tf.data 数据集进行训练和验证

我们已经知道如何处理 loss, metric 和 optimizer，以及当数据为 numpy 数组时如何在 `fit()` 中使用 `validation_data` 和 `validation_split` 参数。

下面看看如何处理 `tf.data.Dataset` 类型的数据。

`tf.data` API 是 TF 2.0 提供的工具，用于以快速、可扩展的方式加载和预处理数据。

有关创建 `Dataset` 的完整指南，可参考 [tf.data 指南](https://www.tensorflow.org/guide/data)。

可以直接将 `Dataset` 实例传递给 `fit()`, `evaluate()` 和 `predict()` 方法：

```python
model = get_compiled_model()

# First, let's create a training Dataset instance.
# For the sake of our example, we'll use the same MNIST data as before.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# Shuffle and slice the dataset.
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

# Now we get a test dataset.
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(64)

# Since the dataset already takes care of batching,
# we don't pass a `batch_size` argument.
model.fit(train_dataset, epochs=3)

# You can also evaluate or predict on a dataset.
print("Evaluate")
result = model.evaluate(test_dataset)
dict(zip(model.metrics_names, result))
```

```txt
Epoch 1/3
782/782 [==============================] - 5s 6ms/step - loss: 0.3239 - sparse_categorical_accuracy: 0.9067
Epoch 2/3
782/782 [==============================] - 4s 6ms/step - loss: 0.1529 - sparse_categorical_accuracy: 0.9533
Epoch 3/3
782/782 [==============================] - 4s 5ms/step - loss: 0.1143 - sparse_categorical_accuracy: 0.9643
Evaluate
157/157 [==============================] - 1s 3ms/step - loss: 0.1554 - sparse_categorical_accuracy: 0.9542
{'loss': 0.1554352343082428, 'sparse_categorical_accuracy': 0.954200029373169}
```

`Dataset` 在每个 epoch 结束后重置，因此可以在下一个 epoch 重用。

如果只想使用 Dataset 特定数目的 batch 进行过训练，可以使用 `steps_per_epoch` 参数指定单个 epoch 训练多少次。

如果这样做，数据集在 epoch 结束时不重置，而是继续处理下一批 batches。数据集最终会耗尽数据。

```python
model = get_compiled_model()

# Prepare the training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

# Only use the 100 batches per epoch (that's 64 * 100 samples)
model.fit(train_dataset, epochs=3, steps_per_epoch=100)
```

```txt
Epoch 1/3
100/100 [==============================] - 1s 6ms/step - loss: 0.7619 - sparse_categorical_accuracy: 0.8112
Epoch 2/3
100/100 [==============================] - 1s 5ms/step - loss: 0.3659 - sparse_categorical_accuracy: 0.8956
Epoch 3/3
100/100 [==============================] - 1s 6ms/step - loss: 0.3059 - sparse_categorical_accuracy: 0.9098
<keras.callbacks.History at 0x2c56a2d8a30>
```

### 使用验证集

`fit()` 的 `validation_data` 参数可以传入 `Dataset` 实例：

```python
model = get_compiled_model()

# Prepare the training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

# Prepare the validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(64)

model.fit(train_dataset, epochs=1, validation_data=val_dataset)
```

```txt
782/782 [==============================] - 6s 6ms/step - loss: 0.3354 - sparse_categorical_accuracy: 0.9038 - val_loss: 0.1960 - val_sparse_categorical_accuracy: 0.9399
<keras.callbacks.History at 0x2c57e015340>
```

在每个 epoch 结尾，模型会迭代验证数据集并计算验证 loss 和验证 metrics。

如果只想对数据集的一定量的 batch 运行验证，可使用 `validation_steps` 参数指定验证次数：

```python
model = get_compiled_model()

# Prepare the training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

# Prepare the validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(64)

model.fit(
    train_dataset,
    epochs=1,
    # Only run validation using the first 10 batches of the dataset
    # using the `validation_steps` argument
    validation_data=val_dataset,
    validation_steps=10,
)
```

```txt
782/782 [==============================] - 5s 6ms/step - loss: 0.3447 - sparse_categorical_accuracy: 0.9023 - val_loss: 0.2796 - val_sparse_categorical_accuracy: 0.9297
<keras.callbacks.History at 0x2c57df156d0>
```

注意，验证集在每次使用后都会重置，所以在不同 epoch 始终使用相同的样本进行验证。

`Dataset` 对象不支持 `validation_split` 参数（从训练数据生成 holdout 集），因为该功能需要索引数据集的所有样本，在 `Dataset` API 无法实现。

## 其它输入格式

除了 NumPy 数组，eager tensor，TF `Dataset`，还可以使用 Pandas dataframe，Python generator 生成的 data/label batch 来训练 Keras 模型。

尤其是 `keras.utils.Sequencee` 类提供了一个简单接口用来构建 Python 数据 geenerator，支持多处理器和 shuffle。

建议：

- 数据较小，内存可容，建议用 NumPy 数组；
- 数据较大，且需要分布式训练，建议用 `Dataset`；
- 数据较大，需要执行许多 TF 无法完成的 Python 端处理，如依赖外部库进行数据加载或预处理，建议用 `Sequence`。

## keras.utils.Sequence 输入

继承 `keras.utils.Sequence` 可以生成包含两个重要功能的 Python generator：

- 支持多处理器
- 支持洗牌（调用 `fit()` 时设置 `shuffle=True`）

`Sequence` 需要实现两个方法：

- `__getitem__`
- `__len__`

`__getitem__` 返回一个完整的 batch。如果要在 epoch 之间修改数据集，则还应该实现 `on_epoch_end`。

例如：

```python
from skimage.io import imread
from skimage.transform import resize
import numpy as np


# Here, `filenames` is list of path to the images
# and `labels` are the associated labels.

class CIFAR10Sequence(Sequence):
    def __init__(self, filenames, labels, batch_size):
        self.filenames, self.labels = filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array([
            resize(imread(filename), (200, 200))
            for filename in batch_x]), np.array(batch_y)


sequence = CIFAR10Sequence(filenames, labels, batch_size)
model.fit(sequence, epochs=10)
```

## 样本加权和类别加权

样本的权重（weight）默认由其在数据集中的频率决定。有两种与样本频率无关的数据加权方法：

- 类别加权（Class weights）
- 样本加权（Sample weights）

### 类别加权

通过将 dict 传递给 `Model.fit()` 的 `class_weight` 参数设置类别加权。该 dict 将类别 index 映射对应类别样本的加权值。

使用过类别加权可用于平衡类别而无需重新采用，也可以在训练模型时偏向特定类别。

例如，如果数据中类别 "0" 是类别 "1" 的一般，则可以使用 `Model.fit(..., class_weight={0: 1., 1: 0.5})`。

下面是一个 NumPy 示例，我们使用类别加权或样本加权来偏向正确的类别 #5 (在 MNIST 数据集中为数字 "5")。

```python
import numpy as np

class_weight = {
    0: 1.0,
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 1.0,
    # Set weight "2" for class "5",
    # making this class 2x more important
    5: 2.0,
    6: 1.0,
    7: 1.0,
    8: 1.0,
    9: 1.0,
}

print("Fit with class weight")
model = get_compiled_model()
model.fit(x_train, y_train, class_weight=class_weight, batch_size=64, epochs=1)
```

```txt
Fit with class weight
782/782 [==============================] - 5s 6ms/step - loss: 0.3833 - sparse_categorical_accuracy: 0.8981
<keras.callbacks.History at 0x2c57e26d610>
```

### 样本加权

为了更细粒度的控制，或者不是构建 classifier，则可以使用样本加权。

- 使用 NumPy 数据进行训练时，将 `sample_weight` 参数传递给 `Model.fit()`；
- 使用 `tf.data` 后其它类似的迭代器训练，yield `(input_batch, label_batch, sample_weight_batch)` 形式的 tuple。

样本加权是一个数组，指定在计算总 loss 时 batch 中每个样本的权重。样本加权通常用在不平衡分类问题种，给罕见类别更大的权重。

当加权值只有 0 和 1，此时的加权数组可以看做 loss 函数的 mask（即完全忽略某些样本对总 loss 的贡献）。

```python
sample_weight = np.ones(shape=(len(y_train),))
sample_weight[y_train == 5] = 2.0

print("Fit with sample weight")
model = get_compiled_model()
model.fit(x_train, y_train, sample_weight=sample_weight, batch_size=64, epochs=1)
```

```txt
Fit with sample weight
782/782 [==============================] - 4s 5ms/step - loss: 0.3836 - sparse_categorical_accuracy: 0.8984
<keras.callbacks.History at 0x2c57e387ca0>
```

下面是对应的 `Dataset` 示例：

```python
sample_weight = np.ones(shape=(len(y_train),))
sample_weight[y_train == 5] = 2.0

# Create a Dataset that includes sample weights
# (3rd element in the return tuple).
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train, sample_weight))

# Shuffle and slice the dataset.
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

model = get_compiled_model()
model.fit(train_dataset, epochs=1)
```

```txt
782/782 [==============================] - 5s 6ms/step - loss: 0.3603 - sparse_categorical_accuracy: 0.9069
<keras.callbacks.History at 0x2c57e49d670>
```

## 多输入/多输出模型的数据

在前面的示例中，模型都是单个输入（`(764,)` 的张量）和单个输出（`(10,)` 预测类别张量）。那么如何处理包含多个输入、多个输出的模型呢？

考虑如下模型，其输入图片 shape 为 `(32, 32, 3)`（对应 `(height, width, channels)`），以及时间序列输入 `(None, 10)`（对应 `(timesteps, features)`）；模型组合这些输入，输出两个值："score" (shape `(1,)`) 和在五个类别上的概率分布（shape `(5,)`）。

```python
image_input = keras.Input(shape=(32, 32, 3), name="img_input")
timeseries_input = keras.Input(shape=(None, 10), name="ts_input")

x1 = layers.Conv2D(3, 3)(image_input)
x1 = layers.GlobalMaxPooling2D()(x1)

x2 = layers.Conv1D(3, 3)(timeseries_input)
x2 = layers.GlobalMaxPooling1D()(x2)

x = layers.concatenate([x1, x2])

score_output = layers.Dense(1, name="score_output")(x)
class_output = layers.Dense(5, name="class_output")(x)

model = keras.Model(
    inputs=[image_input, timeseries_input], outputs=[score_output, class_output]
)
```

绘制模型（注意图中显示的 shape 是 batch shape，而不是样本 shape）：

```python
keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
```

![](2022-09-27-14-24-31.png)

在编译时，传入 loss 函数 list，可以为不同输出指定不同 loss 函数：

```python
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[keras.losses.MeanSquaredError(), keras.losses.CategoricalCrossentropy()],
)
```

如果只提供一个 loss 函数，那么将该 loss 函数应用于每个输出（此处这么做不合适）。

对 metrics 同理：

```python
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[keras.losses.MeanSquaredError(), keras.losses.CategoricalCrossentropy()],
    metrics=[
        [
            keras.metrics.MeanAbsolutePercentageError(),
            keras.metrics.MeanAbsoluteError(),
        ],
        [keras.metrics.CategoricalAccuracy()],
    ],
)
```

由于我们给输出层命名了，所以可以通过 dict 指定每个输出的 loss 和 metrics：

```python
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={
        "score_output": keras.losses.MeanSquaredError(),
        "class_output": keras.losses.CategoricalCrossentropy(),
    },
    metrics={
        "score_output": [
            keras.metrics.MeanAbsolutePercentageError(),
            keras.metrics.MeanAbsoluteError(),
        ],
        "class_output": [keras.metrics.CategoricalAccuracy()],
    },
)
```

当有 2 个以上输出时，建议使用这种显式命名和 dict 的方式。

使用 `loss_weights` 参数可以为不同输出的 losses 指定不同的权重，例如，在上例中我们可以更重视 "score" loss，将其加权设置为 class loss 的 2 倍：

```python
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={
        "score_output": keras.losses.MeanSquaredError(),
        "class_output": keras.losses.CategoricalCrossentropy(),
    },
    metrics={
        "score_output": [
            keras.metrics.MeanAbsolutePercentageError(),
            keras.metrics.MeanAbsoluteError(),
        ],
        "class_output": [keras.metrics.CategoricalAccuracy()],
    },
    loss_weights={"score_output": 2.0, "class_output": 1.0},
)
```

也可以选择不计算某些输出的 loss，比如这些输出纯用于推理、不用于训练：

```python
# List loss version
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[None, keras.losses.CategoricalCrossentropy()],
)

# Or dict loss version
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={"class_output": keras.losses.CategoricalCrossentropy()},
)
```

在 `fit()` 中为多输入、输出模型传入数据与 compile 中指定 loss 函数类似：

- 传入 NumPy 数组列表
- 或输出名称映射到 NumPy 数组的 dict。

```python
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[keras.losses.MeanSquaredError(), keras.losses.CategoricalCrossentropy()],
)

# Generate dummy NumPy data
img_data = np.random.random_sample(size=(100, 32, 32, 3))
ts_data = np.random.random_sample(size=(100, 20, 10))
score_targets = np.random.random_sample(size=(100, 1))
class_targets = np.random.random_sample(size=(100, 5))

# Fit on lists
model.fit([img_data, ts_data], [score_targets, class_targets], batch_size=32, epochs=1)

# Alternatively, fit on dicts
model.fit(
    {"img_input": img_data, "ts_input": ts_data},
    {"score_output": score_targets, "class_output": class_targets},
    batch_size=32,
    epochs=1,
)
```

```txt
4/4 [==============================] - 5s 80ms/step - loss: 13.4527 - score_output_loss: 2.5108 - class_output_loss: 10.9419
4/4 [==============================] - 0s 9ms/step - loss: 12.3336 - score_output_loss: 1.6724 - class_output_loss: 10.6612
<keras.callbacks.History at 0x2c467937490>
```

下面是 `Dataset` 的示例，与 NumPy 数组类似，`Dataset` 应该返回 tuple dict。

```python
train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        {"img_input": img_data, "ts_input": ts_data},
        {"score_output": score_targets, "class_output": class_targets},
    )
)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

model.fit(train_dataset, epochs=1)
```

```txt
2/2 [==============================] - 1s 65ms/step - loss: 11.2755 - score_output_loss: 1.3571 - class_output_loss: 9.9184
<keras.callbacks.History at 0x2c57e5fda90>
```

## callback

Keras 中的 callback 是在训练过程中不同位置调用的对象，如在 epoch 开头和结尾、batch 的开头或结尾等。可用于实现多种功能，例如：

- 训练期间不同位置进行验证（不仅仅是内置的 epoch 结尾验证）
- 定期 checkpoint 模型或当模型超过某个精度时 checkpoint 模型
- 当训练趋于平稳时，改变模型的学习率
- 当训练趋于平稳时，对顶层进行微调
- 在训练结束或超过某个性能阈值时，发送邮件通知
- ...

可以将多个 callbacks 以列表形式传给 `fit()`：

```python
model = get_compiled_model()

callbacks = [
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_loss",
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=2,
        verbose=1,
    )
]
model.fit(
    x_train,
    y_train,
    epochs=20,
    batch_size=64,
    callbacks=callbacks,
    validation_split=0.2,
)
```

```txt
Epoch 1/20
625/625 [==============================] - 4s 7ms/step - loss: 0.3778 - sparse_categorical_accuracy: 0.8952 - val_loss: 0.2279 - val_sparse_categorical_accuracy: 0.9313
Epoch 2/20
625/625 [==============================] - 4s 6ms/step - loss: 0.1743 - sparse_categorical_accuracy: 0.9488 - val_loss: 0.1758 - val_sparse_categorical_accuracy: 0.9464
Epoch 3/20
625/625 [==============================] - 4s 6ms/step - loss: 0.1267 - sparse_categorical_accuracy: 0.9620 - val_loss: 0.1544 - val_sparse_categorical_accuracy: 0.9542
Epoch 4/20
625/625 [==============================] - 4s 6ms/step - loss: 0.0980 - sparse_categorical_accuracy: 0.9700 - val_loss: 0.1570 - val_sparse_categorical_accuracy: 0.9538
Epoch 5/20
625/625 [==============================] - 3s 6ms/step - loss: 0.0818 - sparse_categorical_accuracy: 0.9750 - val_loss: 0.1419 - val_sparse_categorical_accuracy: 0.9610
Epoch 6/20
625/625 [==============================] - 3s 6ms/step - loss: 0.0689 - sparse_categorical_accuracy: 0.9789 - val_loss: 0.1370 - val_sparse_categorical_accuracy: 0.9623
Epoch 7/20
625/625 [==============================] - 3s 6ms/step - loss: 0.0582 - sparse_categorical_accuracy: 0.9826 - val_loss: 0.1453 - val_sparse_categorical_accuracy: 0.9605
Epoch 7: early stopping
<keras.callbacks.History at 0x2c5758178e0>
```

### 内置 callback

Keras 已内置许多 callback，例如：

- `ModelCheckpoint`，周期性的保存模型；
- `EarlyStopping`，当训练不再改善 validation metrics 时终止训练；
- `TensorBoard`，定期输出可以在 TensorBoard 中可视化的模型日志；
- `CSVLogger`，将 loss 和 metrics 数据保存为 CSV 文件。

### 自定义 callback

通过扩展 `keras.callbacks.Callback` 类可以创建自定义 callback。callback 可以通过类属性 `self.model` 访问其关联的模型。

具体如何自定义 callback 可参考[自定义 callback 完整指南](https://tensorflow.google.cn/guide/keras/custom_callback/)。

下面是一个简单示例，保存训练期间的 batch loss 值：

```python
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.per_batch_losses = []

    def on_batch_end(self, batch, logs):
        self.per_batch_losses.append(logs.get("loss"))
```

## Checkpoint 模型

在大型数据集上训练模型时，定期保存模型的 checkpoint 非常重要。

实现该功能的最简单方式是使用 `ModelCheckpoint` callback：

```python
model = get_compiled_model()

callbacks = [
    keras.callbacks.ModelCheckpoint(
        # Path where to save the model
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        # The saved model name will include the current epoch.
        filepath="mymodel_{epoch}",
        save_best_only=True,  # Only save a model if `val_loss` has improved.
        monitor="val_loss",
        verbose=1,
    )
]
model.fit(
    x_train, y_train, epochs=2, batch_size=64, callbacks=callbacks, validation_split=0.2
)
```

```txt
Epoch 1/2
623/625 [============================>.] - ETA: 0s - loss: 0.3714 - sparse_categorical_accuracy: 0.8960
Epoch 1: val_loss improved from inf to 0.23988, saving model to mymodel_1
INFO:tensorflow:Assets written to: mymodel_1\assets
625/625 [==============================] - 5s 7ms/step - loss: 0.3708 - sparse_categorical_accuracy: 0.8962 - val_loss: 0.2399 - val_sparse_categorical_accuracy: 0.9292
Epoch 2/2
620/625 [============================>.] - ETA: 0s - loss: 0.1749 - sparse_categorical_accuracy: 0.9486
Epoch 2: val_loss improved from 0.23988 to 0.17634, saving model to mymodel_2
INFO:tensorflow:Assets written to: mymodel_2\assets
625/625 [==============================] - 4s 6ms/step - loss: 0.1744 - sparse_categorical_accuracy: 0.9488 - val_loss: 0.1763 - val_sparse_categorical_accuracy: 0.9456
<keras.callbacks.History at 0x2c578e43550>
```

`ModelCheckpoint` callback 可用来实现容错性：在训练被随机打断时，能够让模型从上次保存的状态重新开始训练。下面是一个基本示例：

```python
import os

# Prepare a directory to store all the checkpoints.
checkpoint_dir = "./ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return get_compiled_model()


model = make_or_restore_model()
callbacks = [
    # This callback saves a SavedModel every 100 batches.
    # We include the training loss in the saved model name.
    keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir + "/ckpt-loss={loss:.2f}", save_freq=100
    )
]
model.fit(x_train, y_train, epochs=1, callbacks=callbacks)
```

```txt
Creating a new model
  92/1563 [>.............................] - ETA: 8s - loss: 0.9860 - sparse_categorical_accuracy: 0.7272INFO:tensorflow:Assets written to: ./ckpt\ckpt-loss=0.95\assets
 196/1563 [==>...........................] - ETA: 11s - loss: 0.6952 - sparse_categorical_accuracy: 0.8080INFO:tensorflow:Assets written to: ./ckpt\ckpt-loss=0.69\assets
 294/1563 [====>.........................] - ETA: 11s - loss: 0.5810 - sparse_categorical_accuracy: 0.8380INFO:tensorflow:Assets written to: ./ckpt\ckpt-loss=0.58\assets
 399/1563 [======>.......................] - ETA: 10s - loss: 0.5108 - sparse_categorical_accuracy: 0.8555INFO:tensorflow:Assets written to: ./ckpt\ckpt-loss=0.51\assets
 495/1563 [========>.....................] - ETA: 10s - loss: 0.4653 - sparse_categorical_accuracy: 0.8680INFO:tensorflow:Assets written to: ./ckpt\ckpt-loss=0.46\assets
 597/1563 [==========>...................] - ETA: 9s - loss: 0.4356 - sparse_categorical_accuracy: 0.8759INFO:tensorflow:Assets written to: ./ckpt\ckpt-loss=0.43\assets
 691/1563 [============>.................] - ETA: 8s - loss: 0.4103 - sparse_categorical_accuracy: 0.8827INFO:tensorflow:Assets written to: ./ckpt\ckpt-loss=0.41\assets
 789/1563 [==============>...............] - ETA: 7s - loss: 0.3883 - sparse_categorical_accuracy: 0.8881INFO:tensorflow:Assets written to: ./ckpt\ckpt-loss=0.39\assets
 899/1563 [================>.............] - ETA: 6s - loss: 0.3717 - sparse_categorical_accuracy: 0.8927INFO:tensorflow:Assets written to: ./ckpt\ckpt-loss=0.37\assets
 992/1563 [==================>...........] - ETA: 5s - loss: 0.3574 - sparse_categorical_accuracy: 0.8967INFO:tensorflow:Assets written to: ./ckpt\ckpt-loss=0.36\assets
1098/1563 [====================>.........] - ETA: 4s - loss: 0.3451 - sparse_categorical_accuracy: 0.9001INFO:tensorflow:Assets written to: ./ckpt\ckpt-loss=0.35\assets
1199/1563 [======================>.......] - ETA: 3s - loss: 0.3349 - sparse_categorical_accuracy: 0.9029INFO:tensorflow:Assets written to: ./ckpt\ckpt-loss=0.33\assets
1298/1563 [=======================>......] - ETA: 2s - loss: 0.3251 - sparse_categorical_accuracy: 0.9055INFO:tensorflow:Assets written to: ./ckpt\ckpt-loss=0.32\assets
1397/1563 [=========================>....] - ETA: 1s - loss: 0.3169 - sparse_categorical_accuracy: 0.9080INFO:tensorflow:Assets written to: ./ckpt\ckpt-loss=0.32\assets
1499/1563 [===========================>..] - ETA: 0s - loss: 0.3081 - sparse_categorical_accuracy: 0.9102INFO:tensorflow:Assets written to: ./ckpt\ckpt-loss=0.31\assets
1563/1563 [==============================] - 15s 9ms/step - loss: 0.3025 - sparse_categorical_accuracy: 0.9116
<keras.callbacks.History at 0x2c57ebcec70>
```

也可以自定义 callback 来保存和恢复模型。

有关序列化和保存模型的完整指南，可参考 [Save and load Keras models](https://tensorflow.google.cn/guide/keras/save_and_serialize/)。

## 学习率设置

在训练深度学习模型时，一种常见的模式是随着训练的进行逐渐减少学习率，称为 “学习率衰减”（learning rate decay）。

学习率衰减可以是静态的（提前确定，epoch 或 batch index 的函数），或动态的（根据模型当前行为，特别是 validation loss）。

### 在 optimizer 设置预定义 LR

将 schedule 对象以 `learning_rate` 参数传递给 optimizer 设置静态学习率衰减：

```python
initial_learning_rate = 0.1
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)

optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule)
```

Keras 内置有多个 schedules: `ExponentialDecay`, `PiecewiseConstantDecay`, `PolynomialDecay` 以及 `InverseTimeDecay`。

### 使用 callback 实现动态 LR schedule

由于 optimizer 无法访问 validation metrics，所以无法使用上面的 schedule 对象剩下动态 LR schedule（例如，当 validation loss 不再改善时降低 LR）。

但是，callback 可以访问所有 metrics，包括 validation metrics。因此可以通过 callback 来修改 optimizer 上的当前 LR。实际上，内置的 `ReduceLROnPlateau` callback 就实现了该功能。

## 训练期间可视化 loss 和 metrics

在训练期间查看模型性能的最佳方法是使用 TensorBoard，这是一个基于浏览器的应用程序，可以在本地运行，可以：

- 提供训练和评估期间 loss 和 metrics 的实况图
- （可选）可视化 layer 激活的直方图
- （可选）3D 可视化 `Embedding` 层学到的嵌入空间

如果是使用 pip 安装的 TensorFlow，则可以使用如下命令启动 TensorBoard:

```bash
tensorboard --logdir=/full_path_to_your_logs
```

### 使用 TensorBoard callback

在 Keras 模型的 `fit()` 方法中使用 TensorBoard 的最简单方法是使用 `TensorBoard` callback。

最简单的情况下，只需指定 callback 函数写入日志的位置：

```python
keras.callbacks.TensorBoard(
    log_dir="/full_path_to_your_logs",
    histogram_freq=0,  # 记录直方图可视化的频率
    embeddings_freq=0,  # 记录嵌入可视化的频率
    update_freq="epoch", # 写入日志的频率（默认：每个 epoch 一次）
)
```

```txt
<keras.callbacks.TensorBoard at 0x27f5f46f0d0>
```

更多信息，请参考 [TensorBoard callback 文档](https://tensorflow.google.cn/api_docs/python/tf/keras/callbacks/TensorBoard)。

## 参考

- https://www.tensorflow.org/guide/keras/train_and_evaluate
