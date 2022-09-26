# 使用内置方法训练和评估

- [使用内置方法训练和评估](#使用内置方法训练和评估)
  - [简介](#简介)
  - [API 概述](#api-概述)
  - [compile()：指定 loss, metrics 和 optimizr](#compile指定-loss-metrics-和-optimizr)
    - [内置 optimizer, loss 和 metric](#内置-optimizer-loss-和-metric)
    - [自定义 loss](#自定义-loss)
    - [自定义 metric](#自定义-metric)
    - [不符合标准签名的损失和指标](#不符合标准签名的损失和指标)
  - [学习率](#学习率)
  - [训练时可视化损失和指标](#训练时可视化损失和指标)
    - [使用 TensorBoard callback](#使用-tensorboard-callback)
  - [参考](#参考)

***

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

考虑如下模型（这里用的函数 API，使用 Sequential 模型或子类模型亦可）：

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

如果所需 loss 除了 `y_true` 和 `y_pred`，还需要其它参数。则可以扩展 `tf.keras.losses.Loss` 类，并实现以下两个方法：

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
782/782 [==============================] - 5s 5ms/step - loss: 0.3419 - categorical_true_positives: 45148.0000
Epoch 2/3
782/782 [==============================] - 4s 5ms/step - loss: 0.1557 - categorical_true_positives: 47646.0000
Epoch 3/3
782/782 [==============================] - 4s 5ms/step - loss: 0.1166 - categorical_true_positives: 48200.0000
<keras.callbacks.History at 0x27f5f48c1f0>
```

### 不符合标准签名的损失和指标

绝大多数的损失值和指标可以从 `y_true` 和 `y_pred` 计算，其中 `y_pred` 是模型预测结果，但不是全部如此。例如，正则化损失可能只需要激活值，并且该激活值不一定是模型输出。

在这种情况，可以在自定义层的 `call` 方法中调用 `self.add_loss(loss_value)`。以这种方式添加的损失在训练过程中被添加到 "main" 损失中（通过 `compile()`）。例如，添加正则化（Keras 内置有正则化，此处为了演示提供了一个实现）:

```python
class ActivityRegularizationLayer(layers.Layer):
    def call(self, inputs):
        self.add_loss(tf.reduce_sum(inputs) * 0.1)
        return inputs  # Pass-through layer.


inputs = keras.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)

# Insert activity regularization as a layer
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
782/782 [==============================] - 4s 5ms/step - loss: 2.4600
<keras.callbacks.History at 0x27f5f4728e0>
```

可以使用 `add_metric()` 对 logging 指标值做相同操作：

```python

```

## 学习率

在训练深度学习模型时，一种常见的模式是随着训练的进行逐渐减少学习率，称为 “学习率衰减”。

学习率衰减可以是静态的（提前确定，当前 epoch 或当前 batch index 的函数），或动态的（）

## 训练时可视化损失和指标

在训练期间查看模型性能的最佳方法是使用 TensorBoard，这是一个基于浏览器的应用程序，可以在本地运行，提供：

- 提供训练和评估期间损失和指标的实况图
- （可选）可视化 layer 激活的直方图
- （可选）3D 可视化 `Embedding` 层学到的嵌入空间

如果是使用 pip 安装的 TensorFlow，则可以使用如下命令启动 TensorBoard:

```bash
tensorboard --logdir=/full_path_to_your_logs
```

### 使用 TensorBoard callback

在 Keras 模型和 `fit()` 方法中使用 TensorBoard 的最简单方法是使用 `TensorBoard` callback。

最简单的使用方法是，只需指定 callback 函数写入日志的位置：

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

## 参考

- https://www.tensorflow.org/guide/keras/train_and_evaluate
