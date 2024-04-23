# KerasTuner 入门

- [KerasTuner 入门](#kerastuner-入门)
  - [简介](#简介)
  - [tune 模型架构](#tune-模型架构)
    - [定义搜索空间](#定义搜索空间)
    - [搜索](#搜索)
    - [查询结果](#查询结果)
    - [重新训练模型](#重新训练模型)
  - [Tune 模型训练](#tune-模型训练)
  - [Tune 数据预处理](#tune-数据预处理)
    - [重新训练模型](#重新训练模型-1)
  - [指定 tune 目标](#指定-tune-目标)
    - [内置 metric 作为 objective](#内置-metric-作为-objective)
    - [自定义 metric 作为 objective](#自定义-metric-作为-objective)
  - [Tune end-to-end workflows](#tune-end-to-end-workflows)
    - [Tune any function](#tune-any-function)
    - [Keep Keras code separate](#keep-keras-code-separate)
  - [KerasTuner 内置的可调应用：HyperResNett 和 HypeerXception](#kerastuner-内置的可调应用hyperresnett-和-hypeerxception)
  - [参考](#参考)

Last updated: 2022-09-01, 16:46
@author Jiawei Mao
****

## 简介

KerasTuner 是一个通用超参数调优库。它与 Keras 工作流集成很强，但也可用于调优 scikit-learn 模型或其它模型。本教程介绍如何使用 KerasTuner 优化模型架构、训练过程以及数据预处理步骤。从一个简单的示例开始。

## tune 模型架构

首先，定义一个返回编译的 Keras 模型的函数。该函数以 `hp` 为参数，用于在构建模型时定义超参数。

### 定义搜索空间

下面的示例代码定义了包含两个 `Dense` 层的 Keras 模型。我们要 tune 第一个 `Dense` 层的 unit 数。我们只需用 `hp.Int('units', min_value=32, max_value=512, step=32)` 定义一个 integer 超参数，其范围从 32 到 512（inclusive），采样间隔步长最小为 32.

```python
from tensorflow import keras
from tensorflow.keras import layers

def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
    model.add(
        layers.Dense(
            # Define the hyperparameter.
            units=hp.Int("units", min_value=32, max_value=512, step=32),
            activation="relu",
        )
    )
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
```

可以快速测试模型是否构建成功。

```python
import keras_tuner

build_model(keras_tuner.HyperParameters())
```

```txt
<keras.engine.sequential.Sequential at 0x2204482d700>
```

还有许多其它类型的超参数。在函数中可以定义多个超参数，下面，我们用 `hp.Boolean()` tune 是否用 `Dropout` 层，用 `hp.Choice()` tune 使用哪个激活函数，用 `hp.Float()` tune 优化器的学习率。

```python
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
    model.add(
        layers.Dense(
            # Tune number of units.
            units=hp.Int("units", min_value=32, max_value=512, step=32),
            # Tune the activation function to use.
            activation=hp.Choice("activation", ["relu", "tanh"]),
        )
    )
    # Tune whether to use dropout.
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.25))
    model.add(layers.Dense(10, activation="softmax"))
    # Define the optimizer learning rate as a hyperparameter.
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


build_model(keras_tuner.HyperParameters())
```

```txt
<keras.engine.sequential.Sequential at 0x220022d8760>
```

上面使用的超参数是返回实际值的函数。例如，`hp.Int()` 返回 `int` 值。因此，可以把它们放入变量、for 循环或 if 语句。

```python
hp = keras_tuner.HyperParameters()
print(hp.Int("units", min_value=32, max_value=512, step=32))
```

```txt
32
```

也可以提前定义超参数，将 Keras 代码放在单独的函数中。

```python
def call_existing_code(units, activation, dropout, lr):
    model = keras.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(units=units, activation=activation))
    if dropout:
        model.add(layers.Dropout(rate=0.25))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model(hp):
    units = hp.Int("units", min_value=32, max_value=512, step=32)
    activation = hp.Choice("activation", ["relu", "tanh"])
    dropout = hp.Boolean("dropout")
    lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    # call existing model-building code with the hyperparameter values.
    model = call_existing_code(
        units=units, activation=activation, dropout=dropout, lr=lr
    )
    return model


build_model(keras_tuner.HyperParameters())
```

```txt
<keras.engine.sequential.Sequential at 0x220022d8d60>
```

每个超参数根据名称识别（第一个参数）。如果不同的 `Dense` 层的 unit 数使用不同的超参数单独 tune，我们为其分别命名为 `f"units_{i}"`。

值得注意的是，这也是创建条件超参数的例子。例如，`units_3` 仅在 `num_layers` 大于 3 时使用。使用 KerasTuner 可以轻松动态定义此类超参数。

```python
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
    # Tune the number of layers.
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(
            layers.Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                activation=hp.Choice("activation", ["relu", "tanh"]),
            )
        )
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.25))
    model.add(layers.Dense(10, activation="softmax"))
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


build_model(keras_tuner.HyperParameters())
```

```txt
<keras.engine.sequential.Sequential at 0x220022d8cd0>
```

### 搜索

定义搜索空间后，需要选择一个 tuner 类来搜索，如 `RandomSearch`, `BayesianOptimization` 以及 `Hyperband`，对应不同的 tuning 算法。这里我们已 `RandomSearch` 为例。

初始化 tuner 需要指定几个参数：

- `hypermodel`，模型构建函数，本例中为 `build_model`；
- `objective`，优化指标的名称，后面会介绍如何使用自定义指标；
- `max_trials`，搜索期间运行的 trial 次数；
- `executions_per_trial`，每次 trial 构建并拟合的模型数量。不过 trial 具有不同的超参数值。同一个 trial 中的超参数值相同。一次 trial 执行多次是为了减少结果偏差，从而能够更准确地评估模型的性能。如果希望能更快获得结果，可以设置 `executions_per_trial=1`，即每个模型配置只训练一次；
- `overwrite`，是否覆盖同一目录中之前的结果，还是恢复之前的搜索。这里我们设置 `overwrite=True` 启动新的搜索，忽略之前的任何结果；
- `directory`，保存搜索结果的目录；
- `project_name`，`directory` 中子目录名称。

```python
tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=3,
    executions_per_trial=2,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld",
)
```

查看搜索空间摘要：

```python
tuner.search_space_summary()
```

```txt
Search space summary
Default search space size: 5
num_layers (Int)
{'default': None, 'conditions': [], 'min_value': 1, 'max_value': 3, 'step': 1, 'sampling': None}
units_0 (Int)
{'default': None, 'conditions': [], 'min_value': 32, 'max_value': 512, 'step': 32, 'sampling': None}
activation (Choice)
{'default': 'relu', 'conditions': [], 'values': ['relu', 'tanh'], 'ordered': False}
dropout (Boolean)
{'default': False, 'conditions': []}
lr (Float)
{'default': 0.0001, 'conditions': [], 'min_value': 0.0001, 'max_value': 0.01, 'step': None, 'sampling': 'log'}
```

在搜索前，让我们准备一下 MNIST 数据集。

```python
from tensorflow import keras
import numpy as np

(x, y), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x[:-10000]
x_val = x[-10000:]
y_train = y[:-10000]
y_val = y[-10000:]

x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
x_val = np.expand_dims(x_val, -1).astype("float32") / 255.0
x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
```

然后，开始搜索最佳的超参数配置。传递给 `search` 的所有参数在每次执行都传递给了 `model.fit()`。切记设置 `validation_data` 以评估模型。

```python
tuner.search(x_train, y_train, epochs=2, validation_data=(x_val, y_val))
```

```txt
Trial 3 Complete [00h 00m 34s]
val_accuracy: 0.9261500239372253

Best val_accuracy So Far: 0.9612500071525574
Total elapsed time: 00h 01m 42s
INFO:tensorflow:Oracle triggered exit
```

在 `search` 时，在不同 trial 中使用不同的超参数值调用 model-building 函数。在每个 trial，tuner 都会生成一组新的超参数来构建模型，然后对模型进行拟合和评估，记录指标。tuner 逐步搜索超参数空间，最终找到一组好的超参数值。

### 查询结果

搜索结束后，可以查看最佳模型。模型在 `validation_data` 上评估性能最佳的 epoch 保存。

```python
# Get the top 2 models.
models = tuner.get_best_models(num_models=2)
best_model = models[0]
# Build the model.
# Needed for `Sequential` without specified `input_shape`.
best_model.build(input_shape=(None, 28, 28))
best_model.summary()
```

```txt
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 flatten (Flatten)           (None, 784)               0         
                                                                 
 dense (Dense)               (None, 352)               276320    
                                                                 
 dense_1 (Dense)             (None, 256)               90368     
                                                                 
 dropout (Dropout)           (None, 256)               0         
                                                                 
 dense_2 (Dense)             (None, 10)                2570      
                                                                 
=================================================================
Total params: 369,258
Trainable params: 369,258
Non-trainable params: 0
_________________________________________________________________
```

还可以打印搜索结果的摘要：

```python
tuner.results_summary()
```

```txt
Results summary
Results in my_dir\helloworld
Showing 10 best trials
<keras_tuner.engine.objective.Objective object at 0x00000220022EB0A0>
Trial summary
Hyperparameters:
num_layers: 2
units_0: 352
activation: tanh
dropout: True
lr: 0.0004620103367161398
units_1: 256
Score: 0.9612500071525574
Trial summary
Hyperparameters:
num_layers: 2
units_0: 352
activation: relu
dropout: True
lr: 0.009415565762164683
units_1: 32
Score: 0.9576499760150909
Trial summary
Hyperparameters:
num_layers: 2
units_0: 416
activation: tanh
dropout: True
lr: 0.007337423857004189
units_1: 480
Score: 0.9261500239372253
```

在目录 `my_dir/helloworld` 可以找到 logs, checkpoints 等的详细信息，即 `directory/project_name` 目录。

还可以用 TensorBoard 和 HParams 插件可视化 tuning 结果。详情可参考 [Visualize the hyperparameter tuning process](https://keras.io/guides/keras_tuner/visualize_tuning/)。

### 重新训练模型

获得最佳超参数后，可以用完整数据集重新训练模型。

```python
# Get the top 2 hyperparameters.
best_hps = tuner.get_best_hyperparameters(5)
# Build the model with the best hp.
model = build_model(best_hps[0])
# Fit with the entire dataset.
x_all = np.concatenate((x_train, x_val))
y_all = np.concatenate((y_train, y_val))
model.fit(x=x_all, y=y_all, epochs=1)
```

```txt
1875/1875 [==============================] - 8s 4ms/step - loss: 0.3023 - accuracy: 0.9112
<keras.callbacks.History at 0x22013026880>
```

## Tune 模型训练

为了 tune 模型构建过程，需要扩展 `HyperModel` 类，这也便于 hypermodel 的共享和重用。

需要覆盖 `HyperModel.build()` 和 `HyperModel.fit()`，分别 tune 模型构建和训练过程。`HyperModel.build()` 方法功能与 model-building 函数一样，即用超参数创建并返回 Keras 模型。

在 `HyperModel.fit()` 中，可以访问 `HyperModel.build()` 返回的模型，`hp` 以及传递给 `search()` 的所有参数。在其中需要训练模型并返回训练的历史记录。

在下面的代码中，我们要 tune `model.fit()` 的 `shuffle` 参数。

epochs 数通常不需要 tune，因为传递给 `model.fit()` 的内置 callback 可以保存基于 `validation_data` 性能最佳 epoch 时的模型。

> **[!NOTE]**
> 必须记得把 `**kwargs` 传递给 `model.fit()`，因为它包含保存模型和 tensorboard 插件的 callbacks。

```python
class MyHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        model = keras.Sequential()
        model.add(layers.Flatten())
        model.add(
            layers.Dense(
                units=hp.Int("units", min_value=32, max_value=512, step=32),
                activation="relu",
            )
        )
        model.add(layers.Dense(10, activation="softmax"))
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            # Tune whether to shuffle the data in each epoch.
            shuffle=hp.Boolean("shuffle"),
            **kwargs,
        )
```

同样，可以快速检查代码是否能正常工作：

```python
hp = keras_tuner.HyperParameters()
hypermodel = MyHyperModel()
model = hypermodel.build(hp)
hypermodel.fit(hp, model, np.random.rand(100, 28, 28), np.random.rand(100, 10))
```

```txt
4/4 [==============================] - 0s 5ms/step - loss: 12.6506 - accuracy: 0.1100
<keras.callbacks.History at 0x220766a43d0>
```

## Tune 数据预处理

要 tune 数据预处理步骤，只需在 `HyperModel.fit()` 中添加一个额外步骤，数据集可以从参数中访问。在下面代码，tune 是否在训练模型之前将数据归一化。这次，我们将 `x` 和 `y` 显式地放在函数签名中，因为我们需要使用。

```python
class MyHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        model = keras.Sequential()
        model.add(layers.Flatten())
        model.add(
            layers.Dense(
                units=hp.Int("units", min_value=32, max_value=512, step=32),
                activation="relu",
            )
        )
        model.add(layers.Dense(10, activation="softmax"))
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, hp, model, x, y, **kwargs):
        if hp.Boolean("normalize"):
            x = layers.Normalization()(x)
        return model.fit(
            x,
            y,
            # Tune whether to shuffle the data in each epoch.
            shuffle=hp.Boolean("shuffle"),
            **kwargs,
        )


hp = keras_tuner.HyperParameters()
hypermodel = MyHyperModel()
model = hypermodel.build(hp)
hypermodel.fit(hp, model, np.random.rand(100, 28, 28), np.random.rand(100, 10))
```

```txt
4/4 [==============================] - 0s 4ms/step - loss: 12.7305 - accuracy: 0.1300
<keras.callbacks.History at 0x2207636cf70>
```

如果某个超参数在 `build()` 和 `fit()` 中都要使用，则可以在 `build()` 中定义，在 `fit()` 中可以用 `hp.get(hp_name)` 获得。以图像 size 为例，它既作为 `build()` 的输入 shape，在 `fit()` 的数据预处理中也要使用。

```python
class MyHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        image_size = hp.Int("image_size", 10, 28)
        inputs = keras.Input(shape=(image_size, image_size))
        outputs = layers.Flatten()(inputs)
        outputs = layers.Dense(
            units=hp.Int("units", min_value=32, max_value=512, step=32),
            activation="relu",
        )(outputs)
        outputs = layers.Dense(10, activation="softmax")(outputs)
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, hp, model, x, y, validation_data=None, **kwargs):
        if hp.Boolean("normalize"):
            x = layers.Normalization()(x)
        image_size = hp.get("image_size")
        cropped_x = x[:, :image_size, :image_size, :]
        if validation_data:
            x_val, y_val = validation_data
            cropped_x_val = x_val[:, :image_size, :image_size, :]
            validation_data = (cropped_x_val, y_val)
        return model.fit(
            cropped_x,
            y,
            # Tune whether to shuffle the data in each epoch.
            shuffle=hp.Boolean("shuffle"),
            validation_data=validation_data,
            **kwargs,
        )


tuner = keras_tuner.RandomSearch(
    MyHyperModel(),
    objective="val_accuracy",
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="tune_hypermodel",
)

tuner.search(x_train, y_train, epochs=2, validation_data=(x_val, y_val))
```

```txt
Trial 3 Complete [00h 00m 12s]
val_accuracy: 0.3244999945163727

Best val_accuracy So Far: 0.9646999835968018
Total elapsed time: 00h 00m 37s
INFO:tensorflow:Oracle triggered exit
```

### 重新训练模型

使用 `HyperModel` 也可以重新训练最佳模型。

```python
hypermodel = MyHyperModel()
best_hp = tuner.get_best_hyperparameters()[0]
model = hypermodel.build(best_hp)
hypermodel.fit(best_hp, model, x_all, y_all, epochs=1)
```

```txt
1875/1875 [==============================] - 6s 3ms/step - loss: 0.2710 - accuracy: 0.9230
<keras.callbacks.History at 0x2203ea33f10>
```

## 指定 tune 目标

上面的所有示例，都使用验证精度 `val_accuracy` 作为 tune 目标来选择最佳模型。实际上，我们可以使用任何指标作为目标，例如 `val_loss` 最常用，即验证损失。

### 内置 metric 作为 objective

Keras 中有许多内置指标可作为目标，[内置指标列表](https://keras.io/api/metrics/)。

将内置指标作为目标，按如下步骤进行：

- 使用内置指标编译模型。例如，如果使用 `MeanAbsoluteError()`，需要使用 `metrics=[MeanAbsoluteError()]` 编译模型，当然也可以使用名称指定 `metrics=["mean_absolute_error"]`。metric 的字符串名称是类名的 snake 形式。
- 确定目标的字符串名称。目标的字符串名称格式为 `f"val_{metric_name_string}"`。例如，在验证数据集上评估的 mean absolute error 的目标名称为 `"val_mean_absolute_error"`。
- 包装为 `keras_tuner.Objective`。通常需要将目标包装到 `keras_tuner.Objective` 对象，以指定优化目标的方向。例如，为了最小化 mean absolute error，可以使用 `keras_tuner.Objective("val_mean_absolute_error", "min")`。优化方向为 `"min"` 或 `"max"`。
- 将 `keras_tuner.Objective` 传递给 tuner。

代码示例：

```python
def build_regressor(hp):
    model = keras.Sequential(
        [
            layers.Dense(units=hp.Int("units", 32, 128, 32), activation="relu"),
            layers.Dense(units=1),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        # Objective is one of the metrics.
        metrics=[keras.metrics.MeanAbsoluteError()],
    )
    return model


tuner = keras_tuner.RandomSearch(
    hypermodel=build_regressor,
    # The objective name and direction.
    # Name is the f"val_{snake_case_metric_class_name}".
    objective=keras_tuner.Objective("val_mean_absolute_error", direction="min"),
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="built_in_metrics",
)

tuner.search(
    x=np.random.rand(100, 10),
    y=np.random.rand(100, 1),
    validation_data=(np.random.rand(20, 10), np.random.rand(20, 1)),
)

tuner.results_summary()
```

```txt
Trial 3 Complete [00h 00m 01s]
val_mean_absolute_error: 0.6171705722808838

Best val_mean_absolute_error So Far: 0.3675425052642822
Total elapsed time: 00h 00m 02s
INFO:tensorflow:Oracle triggered exit
Results summary
Results in my_dir\built_in_metrics
Showing 10 best trials
<keras_tuner.engine.objective.Objective object at 0x0000022076369400>
Trial summary
Hyperparameters:
units: 96
Score: 0.3675425052642822
Trial summary
Hyperparameters:
units: 64
Score: 0.5847377777099609
Trial summary
Hyperparameters:
units: 32
Score: 0.6171705722808838
```

### 自定义 metric 作为 objective

可以将自定义 metric 作为超参数搜索的 objective。下面以 mean squared error (MSE) 为例。首先通过扩展 `keras.metrics.Metric` 实现 MSE metric。记得使用 `super().__init__()` 的 `name` 参数为 metric 指定名称，稍后要使用。注意：MSE 实际上是一个内置 metric，用 `keras.metrics.MeanSquaredError` 导入。这里只是为了演示如何自定义 metric 作为超参数搜索的 objective。

有关自定义 Metric 的更多信息，可参考 [Creating custom metrics](https://keras.io/api/metrics/#creating-custom-metrics)。如果需要一个与 `update_state(y_true, y_pred, sample_weight)` 方法签名不同的 metric，可以按照 [Customizing what happens in fit()](https://keras.io/guides/customizing_what_happens_in_fit/#going-lowerlevel) 覆盖模型的 `train_step()` 方法实现。

```python
import tensorflow as tf


class CustomMetric(keras.metrics.Metric):
    def __init__(self, **kwargs):
        # Specify the name of the metric as "custom_metric".
        super().__init__(name="custom_metric", **kwargs)
        self.sum = self.add_weight(name="sum", initializer="zeros")
        self.count = self.add_weight(name="count", dtype=tf.int32, initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.math.squared_difference(y_pred, y_true)
        count = tf.shape(y_true)[0]
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            values *= sample_weight
            count *= sample_weight
        self.sum.assign_add(tf.reduce_sum(values))
        self.count.assign_add(count)

    def result(self):
        return self.sum / tf.cast(self.count, tf.float32)

    def reset_states(self):
        self.sum.assign(0)
        self.count.assign(0)
```

使用自定义 objective 进行搜索：

```python
def build_regressor(hp):
    model = keras.Sequential(
        [
            layers.Dense(units=hp.Int("units", 32, 128, 32), activation="relu"),
            layers.Dense(units=1),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        # Put custom metric into the metrics.
        metrics=[CustomMetric()],
    )
    return model


tuner = keras_tuner.RandomSearch(
    hypermodel=build_regressor,
    # Specify the name and direction of the objective.
    objective=keras_tuner.Objective("val_custom_metric", direction="min"),
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="custom_metrics",
)

tuner.search(
    x=np.random.rand(100, 10),
    y=np.random.rand(100, 1),
    validation_data=(np.random.rand(20, 10), np.random.rand(20, 1)),
)

tuner.results_summary()
```

```txt
Trial 3 Complete [00h 00m 01s]
val_custom_metric: 0.28978320956230164

Best val_custom_metric So Far: 0.28978320956230164
Total elapsed time: 00h 00m 02s
INFO:tensorflow:Oracle triggered exit
Results summary
Results in my_dir\custom_metrics
Showing 10 best trials
<keras_tuner.engine.objective.Objective object at 0x00000220764EE760>
Trial summary
Hyperparameters:
units: 64
Score: 0.28978320956230164
Trial summary
Hyperparameters:
units: 96
Score: 0.33249369263648987
Trial summary
Hyperparameters:
units: 32
Score: 0.3578064739704132
```

如果自定义 objective 难以放入自定义的 metric，也可以在 `HyperModel.fit()` 中自己评估模型，并返回 objective 值。该 objective 值默认将最小化。此时，在初始化 tuner 时不需要指定 `objective`。这种方法的缺点是，Keras logs 无法记录 metic 值，因此任何 TensorBoard 无法可视化这些值。

```python
class HyperRegressor(keras_tuner.HyperModel):
    def build(self, hp):
        model = keras.Sequential(
            [
                layers.Dense(units=hp.Int("units", 32, 128, 32), activation="relu"),
                layers.Dense(units=1),
            ]
        )
        model.compile(
            optimizer="adam",
            loss="mean_squared_error",
        )
        return model

    def fit(self, hp, model, x, y, validation_data, **kwargs):
        model.fit(x, y, **kwargs)
        x_val, y_val = validation_data
        y_pred = model.predict(x_val)
        # Return a single float to minimize.
        return np.mean(np.abs(y_pred - y_val))


tuner = keras_tuner.RandomSearch(
    hypermodel=HyperRegressor(),
    # No objective to specify.
    # Objective is the return value of `HyperModel.fit()`.
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="custom_eval",
)
tuner.search(
    x=np.random.rand(100, 10),
    y=np.random.rand(100, 1),
    validation_data=(np.random.rand(20, 10), np.random.rand(20, 1)),
)

tuner.results_summary()
```

```txt
Trial 3 Complete [00h 00m 01s]
default_objective: 0.7787678093363477

Best default_objective So Far: 0.32630745469849354
Total elapsed time: 00h 00m 02s
INFO:tensorflow:Oracle triggered exit
Results summary
Results in my_dir\custom_eval
Showing 10 best trials
<keras_tuner.engine.objective.DefaultObjective object at 0x0000022076566E50>
Trial summary
Hyperparameters:
units: 96
Score: 0.32630745469849354
Trial summary
Hyperparameters:
units: 64
Score: 0.7787678093363477
Trial summary
Hyperparameters:
units: 32
Score: 0.8316919580279796
```

如果需要在 KerasTuner 中跟踪多个 metrics，但只使用其中一个作为 objective，则可以返回一个 dict，其 keys 为 metric 名称，values 为 metric 值，例如返回 `{"metric_a": 1.0, "metric_b", 2.0}`。使用其中一个 key 作为 objective 名称，例如 `keras_tuner.Objective("metric_a", "min")`。

```python
class HyperRegressor(keras_tuner.HyperModel):
    def build(self, hp):
        model = keras.Sequential(
            [
                layers.Dense(units=hp.Int("units", 32, 128, 32), activation="relu"),
                layers.Dense(units=1),
            ]
        )
        model.compile(
            optimizer="adam",
            loss="mean_squared_error",
        )
        return model

    def fit(self, hp, model, x, y, validation_data, **kwargs):
        model.fit(x, y, **kwargs)
        x_val, y_val = validation_data
        y_pred = model.predict(x_val)
        # Return a dictionary of metrics for KerasTuner to track.
        return {
            "metric_a": -np.mean(np.abs(y_pred - y_val)),
            "metric_b": np.mean(np.square(y_pred - y_val)),
        }


tuner = keras_tuner.RandomSearch(
    hypermodel=HyperRegressor(),
    # Objective is one of the keys.
    # Maximize the negative MAE, equivalent to minimize MAE.
    objective=keras_tuner.Objective("metric_a", "max"),
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="custom_eval_dict",
)
tuner.search(
    x=np.random.rand(100, 10),
    y=np.random.rand(100, 1),
    validation_data=(np.random.rand(20, 10), np.random.rand(20, 1)),
)

tuner.results_summary()
```

```txt
Trial 3 Complete [00h 00m 01s]
metric_a: -0.44392925086050117

Best metric_a So Far: -0.44392925086050117
Total elapsed time: 00h 00m 02s
INFO:tensorflow:Oracle triggered exit
Results summary
Results in my_dir\custom_eval_dict
Showing 10 best trials
<keras_tuner.engine.objective.Objective object at 0x0000022076461520>
Trial summary
Hyperparameters:
units: 64
Score: -0.44392925086050117
Trial summary
Hyperparameters:
units: 96
Score: -0.46160490621228545
Trial summary
Hyperparameters:
units: 32
Score: -0.5679751862900692
```

## Tune end-to-end workflows

在某些情况，无法将代码拆分为 build 和 fit 函数两部分。此时可以覆盖 `Tuner.run_trial()`，在其中包含 end-to-end workflow，完整控制一次 trial。可以将其视为一个 black-box optimizer。

### Tune any function

例如，查找最小化 `f(x)=x*x+1` 的 `x` 值。下面定义一个 `x` 作为超参数，返回 `f(x)` 作为 objective 值。初始化 tuner 的 `hypermodel` 和 `objective` 参数可以忽略。

```python
class MyTuner(keras_tuner.RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        # Get the hp from trial.
        hp = trial.hyperparameters
        # Define "x" as a hyperparameter.
        x = hp.Float("x", min_value=-1.0, max_value=1.0)
        # Return the objective value to minimize.
        return x * x + 1


tuner = MyTuner(
    # No hypermodel or objective specified.
    max_trials=20,
    overwrite=True,
    directory="my_dir",
    project_name="tune_anything",
)

# No need to pass anything to search()
# unless you use them in run_trial().
tuner.search()
print(tuner.get_best_hyperparameters()[0].get("x"))
```

```txt
Trial 20 Complete [00h 00m 00s]
default_objective: 1.1897559512891216

Best default_objective So Far: 1.002395731331175
Total elapsed time: 00h 00m 00s
INFO:tensorflow:Oracle triggered exit
0.04894620854749632
```

### Keep Keras code separate

可以保持所有 Keras 代码不变，并使用 KerasTuner 对其进行微调。如果由于某种原因无法修改 Keras 代码，该选项非常有用。

这种方式提供了更大的灵活性，不必将模型构建和训练的代码分开。但是，这种 workflow 不会帮助你保存模型或连接 TensorBoard 插件。

要保存模型，可以使用 `trial.trial_id` (用于识别 trial 的字符串) 来创建不同的路径，来保存不同 trials 的模型。

```python
import os


def keras_code(units, optimizer, saving_path):
    # Build model
    model = keras.Sequential(
        [
            layers.Dense(units=units, activation="relu"),
            layers.Dense(units=1),
        ]
    )
    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
    )

    # Prepare data
    x_train = np.random.rand(100, 10)
    y_train = np.random.rand(100, 1)
    x_val = np.random.rand(20, 10)
    y_val = np.random.rand(20, 1)

    # Train & eval model
    model.fit(x_train, y_train)

    # Save model
    model.save(saving_path)

    # Return a single float as the objective value.
    # You may also return a dictionary
    # of {metric_name: metric_value}.
    y_pred = model.predict(x_val)
    return np.mean(np.abs(y_pred - y_val))


class MyTuner(keras_tuner.RandomSearch):
    def run_trial(self, trial, **kwargs):
        hp = trial.hyperparameters
        return keras_code(
            units=hp.Int("units", 32, 128, 32),
            optimizer=hp.Choice("optimizer", ["adam", "adadelta"]),
            saving_path=os.path.join("/tmp", trial.trial_id),
        )


tuner = MyTuner(
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="keep_code_separate",
)
tuner.search()
# Retraining the model
best_hp = tuner.get_best_hyperparameters()[0]
keras_code(**best_hp.values, saving_path="/tmp/best_model")
```

```txt
Trial 3 Complete [00h 00m 00s]
default_objective: 0.4348094390943946

Best default_objective So Far: 0.23868455769793312
Total elapsed time: 00h 00m 03s
INFO:tensorflow:Oracle triggered exit
4/4 [==============================] - 0s 3ms/step - loss: 0.1520
INFO:tensorflow:Assets written to: /tmp/best_model/assets

0.2114115606885921
```

## KerasTuner 内置的可调应用：HyperResNett 和 HypeerXception

这俩都是用于计算机视觉的现成模型。

都以 `loss="categorical_crossentropy"` 和 `metrics=["accuracy"]` 预编译。

```python
from keras_tuner.applications import HyperResNet

hypermodel = HyperResNet(input_shape=(28, 28, 1), classes=10)

tuner = keras_tuner.RandomSearch(
    hypermodel,
    objective="val_accuracy",
    max_trials=2,
    overwrite=True,
    directory="my_dir",
    project_name="built_in_hypermodel",
)

tuner.search(
    x_train[:100], y_train[:100], epochs=1, validation_data=(x_val[:100], y_val[:100])
)
```

```txt
Trial 2 Complete [00h 00m 28s]
val_accuracy: 0.10000000149011612

Best val_accuracy So Far: 0.10000000149011612
Total elapsed time: 00h 00m 50s
INFO:tensorflow:Oracle triggered exit
```

## 参考

- https://keras.io/guides/keras_tuner/getting_started/
