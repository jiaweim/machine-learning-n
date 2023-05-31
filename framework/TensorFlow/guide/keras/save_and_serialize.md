# Keras 模型的保存和加载

- [Keras 模型的保存和加载](#keras-模型的保存和加载)
  - [简介](#简介)
  - [如何保存和加载模型](#如何保存和加载模型)
  - [设置](#设置)
  - [保存和加载整个模型](#保存和加载整个模型)
    - [SavedModel 格式](#savedmodel-格式)
    - [Keras H5 格式](#keras-h5-格式)
  - [保存架构](#保存架构)
    - [Functional 或 Sequential 模型的架构](#functional-或-sequential-模型的架构)
    - [自定义对象](#自定义对象)
  - [仅保存和加载模型的权重](#仅保存和加载模型的权重)
    - [内存中权重迁移 API](#内存中权重迁移-api)
    - [权重到文件的保存和加载 API](#权重到文件的保存和加载-api)
    - [TF Checkpoint 格式](#tf-checkpoint-格式)
    - [HDF5 格式](#hdf5-格式)
  - [参考](#参考)

Last updated: 2023-02-17, 10:03
****

## 简介

Keras 模型由多个组件组成：

- 模型架构/配置，指模型包含哪些 layer，以及这些 layer 的连接方式；
- 权重值（模型状态，state of the model）；
- 优化器（optimizer），通过 `compile` 设置；
- losses 和 metrics，通过 `compile` 或调用 `add_loss()`, `add_metric()` 设置。

Keras API 支持一次性保存所有这些组件，也可以只保存一部分：

- 将所有组件保存为单个 TF SavedModel 归档格式或老式的 Keras H5 格式，这是标准做法；
- 只保存模型架构，通常保存为 JSON 格式；
- 只保存权重，一般在训练模型时使用。

下面依次查看这些选项，以及如何使用它们。

## 如何保存和加载模型

如果你急需知道如何保存和恢复模型，则可以使用下面的方法。

**保存 Keras 模型：**

```python
model = ...  # Get model (Sequential, Functional Model, or Model subclass)
model.save('path/to/location')
```

**重新加载模型：**

```python
from tensorflow import keras
model = keras.models.load_model('path/to/location')
```

下面介绍技术细节。

## 设置

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
```

## 保存和加载整个模型

可以将整个模型保存为单个构建（artifact）。包括：

- 模型架构
- 模型权重
- 模型编译信息
- optimizer 及其状态（便于从中断的地方重新开始训练）

**APIs**

- `model.save()` 或 [tf.keras.models.save_model()](https://www.tensorflow.org/api_docs/python/tf/keras/models/save_model)
- [tf.keras.models.load_model()](https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model)

保存整个模型有两种格式可选：TF SavedModel 格式，以及老的 Keras **H5** 格式。SavedModel 是推荐格式，为 `model.save()` 的默认选项。

切换到 H5 格式的方法：

- `save()` 中设置 `save_format='h5'`
- `save()` 中传入的文件名以 `.h5` 或 `.keras` 结尾

### SavedModel 格式

SavedModel 是一种全面的格式，可以保存模型架构、权重以及函数调用的 TF subgraph。使得 Keras 能够恢复内置 layer 和自定义对象。例如：

```python
def get_model():
    # 创建一个简单的模型
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


model = get_model()

# 训练模型
test_input = np.random.random((128, 32))
test_target = np.random.random((128, 1))
model.fit(test_input, test_target)

# 调用 `save('my_model')` 创建一个 SavedModel 目录 `my_model`.
model.save("my_model")

# 重建相同的模型
reconstructed_model = keras.models.load_model("my_model")

# 检查
np.testing.assert_allclose(
    model.predict(test_input), reconstructed_model.predict(test_input)
)

# 重建的模型已编译并保存了 optimizer 状态，因此可以继续训练：
reconstructed_model.fit(test_input, test_target)
```

```txt
4/4 [==============================] - 1s 2ms/step - loss: 0.5401
INFO:tensorflow:Assets written to: my_model\assets
4/4 [==============================] - 0s 1000us/step
4/4 [==============================] - 0s 1ms/step
4/4 [==============================] - 0s 2ms/step - loss: 0.4676
<keras.callbacks.History at 0x18115d0e7f0>
```

**SavedModel 包含的内容**

调用 `model.save('my_model')` 会创建一个名为 `my_model` 的目录，该目录包含：

```python
!ls my_model
```

```txt
assets keras_metadata.pb  saved_model.pb  variables
```

模型架构和训练参数（包括 optimizer, losses 和 metrics）保存在 `saved_model.pb` 中。权重保存在 `variables/` 目录。

有关 SavedModel 格式的详细信息，可参考 [Using the SavedModel format](https://www.tensorflow.org/guide/saved_model)。

**SavedModel 如何处理自定义对象**

保存模型及其 layers 时，SavedModel 格式保存类名、调用函数、losses 以及 weights。调用函数定义了模型或 layer 的计算图。

在没有 model/layer 配置/架构的情况下，调用函数可用来创建一个类似原始模型的模型，该模型可以训练、评估和推理。

然而，在自定义模型或 layer 时，最好定义 `get_config` 和 `from_config` 方法。以便在需要时更容易更新计算。

例如：

```python
class CustomModel(keras.Model):
    def __init__(self, hidden_units):
        super(CustomModel, self).__init__()
        self.hidden_units = hidden_units
        self.dense_layers = [keras.layers.Dense(u) for u in hidden_units]

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return x

    def get_config(self):
        return {"hidden_units": self.hidden_units}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


model = CustomModel([16, 16, 10])
# Build the model by calling it
input_arr = tf.random.uniform((1, 5))
outputs = model(input_arr)
model.save("my_model")

# Option 1: Load with the custom_object argument.
loaded_1 = keras.models.load_model(
    "my_model", custom_objects={"CustomModel": CustomModel}
)

# Option 2: Load without the CustomModel class.

# Delete the custom-defined model class to ensure that the loader does not have
# access to it.
del CustomModel

loaded_2 = keras.models.load_model("my_model")
np.testing.assert_allclose(loaded_1(input_arr), outputs)
np.testing.assert_allclose(loaded_2(input_arr), outputs)

print("Original model:", model)
print("Model Loaded with custom objects:", loaded_1)
print("Model loaded without the custom object class:", loaded_2)
```

```txt
INFO:tensorflow:Assets written to: my_model\assets
WARNING:tensorflow:No training configuration found in save file, so the model was *not* compiled. Compile it manually.
WARNING:tensorflow:No training configuration found in save file, so the model was *not* compiled. Compile it manually.
Original model: <__main__.CustomModel object at 0x000001810A62DB20>
Model Loaded with custom objects: <__main__.CustomModel object at 0x000001813EDEBAF0>
Model loaded without the custom object class: <keras.saving.saved_model.load.CustomModel object at 0x000001813EDEBDF0>
```

第一个模型使用 config 和 `CustomModel` 类加载，第二个模型通过动态创建于原始模型类似的模型类来加载。

**SavedModel 配置**

*New in TensorFflow 2.4* `model.save` 添加了 `save_traces` 参数，用来切换 SavedModel 的函数 tracing。保存函数是为了让 Keras 在没有原始类定义的情况下重新加载自定义对象，因此当 `save_traces=False` 时，所有自定义对象必须定义 `get_config` 和 `from_config` 方法。加载时，自定义对象必须通过 `custom_objects` 参数指定。`save_traces=False` 减少了 SavedModel 使用的磁盘空间和保存时间。

### Keras H5 格式

Keras 还支持将模型保存为单个 HDF5 文件，包含模型的架构、权重值，以及 `compile()` 信息。它是 SavedModel 的轻量级替代。

```python
model = get_model()

# Train the model.
test_input = np.random.random((128, 32))
test_target = np.random.random((128, 1))
model.fit(test_input, test_target)

# Calling `save('my_model.h5')` creates a h5 file `my_model.h5`.
model.save("my_h5_model.h5")

# It can be used to reconstruct the model identically.
reconstructed_model = keras.models.load_model("my_h5_model.h5")

# Let's check:
np.testing.assert_allclose(
    model.predict(test_input), reconstructed_model.predict(test_input)
)

# The reconstructed model is already compiled and has retained the optimizer
# state, so training can resume:
reconstructed_model.fit(test_input, test_target)
```

```txt
4/4 [==============================] - 0s 3ms/step - loss: 0.5780
4/4 [==============================] - 0s 2ms/step
4/4 [==============================] - 0s 2ms/step
4/4 [==============================] - 0s 3ms/step - loss: 0.5046
<keras.callbacks.History at 0x1816e2db910>
```

**H5 的局限性**

与 SavedModel 格式相比，H5 文件没有包含两个内容：

- 通过 `model.add_loss()` 和 `model.add_metric()` 添加的外部 losses 和 metrics。如果模型包含这类 losses 和 metrics，并且想恢复训练，就需要在加载模型后手动将这些 losses 加回来。不过，这不适用于通过 `self.add_loss()` 和 `self.add_metrics()` 在 layer 内创建的 losses/metrics。只要加载 layer，这类 losses 和 metrics 就被保留，因为它们是 layer `call` 方法的一部分。
- 自定义对象的计算图。加载时，Keras 需要访问这些对象的 Python 类/函数，以便重建模型。

## 保存架构

模型架构（或配置）指模型包含的 layers，以及这些 layers 的连接方式。如果已有模型的架构，那么重新初始化权重就能创建模型，不需要编译信息。

> **Note:** 
> 只适用于使用函数 API 或 Sequential API 创建的模型，subclass 模型不行。

### Functional 或 Sequential 模型的架构

这类模型都是清晰的 layer graph，它们的架构通常为结构化形式。

**API**

- `get_config()` 和 `from_config()`
- `tf.keras.models.model_to_json()` 和 `tf.keras.models.model_from_json()`

**get_config() 和 from_config()**

调用 `config = model.get_config()` 返回包含模型配置（架构）的 Python dict。使用 `Sequential.from_config(config)` （对 `Sequential` 模型）或 `Model.from_config(config)`（对函数 API）可以重建相同的模型。

该工作流适用于任何可序列化 layer。

**Layer 示例**

```python
layer = keras.layers.Dense(3, activation="relu")
layer_config = layer.get_config()
new_layer = keras.layers.Dense.from_config(layer_config)
```

**Sequential 模型示例**

```python
model = keras.Sequential([keras.Input((32,)), keras.layers.Dense(1)])
config = model.get_config()
new_model = keras.Sequential.from_config(config)
```

**Functional 模型示例**

```python
inputs = keras.Input((32,))
outputs = keras.layers.Dense(1)(inputs)
model = keras.Model(inputs, outputs)
config = model.get_config()
new_model = keras.Model.from_config(config)
```

**to_json() 和 `tf.keras.models.model_from_json()`**

与 `get_config` / `from_config` 类似，只是将模型转换为 JSON 字符串，然后可以在不使用原始模型类的情况下加载。该方法用于模型，不适用于 layer。

例如：

```python
model = keras.Sequential([keras.Input((32,)), keras.layers.Dense(1)])
json_config = model.to_json()
new_model = keras.models.model_from_json(json_config)
```

### 自定义对象

**Models and layers**

subclass 模型和 layer 的架构定义在 `__init__` 和 `call` 方法中。它们被当作 Python 字节码处理，不能被序列化为 JSON 兼容的 config，虽然可以尝试序列化字节码（如 `pickle`），但是这种方式不安全，同时意味着模型不能在其它系统上加载。

为了保存、加载包含自定义 layer 的模型，或 subclass 模型，应该覆盖 `get_config` 和 `from_config`（可选）方法。此外，还应该注册自定义对象，使 Keras 知道它。

**自定义函数**

自定义函数，如激活 loss 或初始化函数，不需要 `get_config` 方法。将函数注册为自定义对象后，使用函数名就足以加载。

**只加载 TF graph**

可以加载 Keras 生成的 TF graph。不需要提供任何 `custom_objects`，如下：

```python
model.save("my_model")
tensorflow_graph = tf.saved_model.load("my_model")
x = np.random.uniform(size=(4, 32)).astype(np.float32)
predicted = tensorflow_graph(x).numpy()
```

```txt
WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
INFO:tensorflow:Assets written to: my_model\assets
```

注意，该方法有几个缺点：

- 出于可追溯性的原因，应该始终可以访问使用的自定义对象。将无法重新创建的模型投入生产不合适。
- `tf.saved_model.load` 返回的对象不是 Keras 模型。所以它不怎么好用，比如，不能访问 `.predict()` 或 `.fit()` 方法。

即使不推荐使用，在特殊情况下，比如丢失了自定义对象的代码，或使用 `tf.keras.models.load_model()` 加载模型遇到问题，该方法有助于解决问题。

更多有关 [tf.saved_model.load](https://tensorflow.google.cn/api_docs/python/tf/saved_model/load) 的信息。

**定义 config 方法**

规范：

- `get_config` 应该返回一个可 JSON 序列化的 dict，以便与 Keras 架构和模型保存 API 兼容；
- `from_config(config)` (`classmethod`)应该返回一个从 config 创建的新的 layer 或 model。默认实现返回 `cls(**config)`。

例如：

```python
class CustomLayer(keras.layers.Layer):
    def __init__(self, a):
        self.var = tf.Variable(a, name="var_a")

    def call(self, inputs, training=False):
        if training:
            return inputs * self.var
        else:
            return inputs

    def get_config(self):
        return {"a": self.var.numpy()}

    # There's actually no need to define `from_config` here, since returning
    # `cls(**config)` is the default behavior.
    @classmethod
    def from_config(cls, config):
        return cls(**config)


layer = CustomLayer(5)
layer.var.assign(2)

serialized_layer = keras.layers.serialize(layer)
new_layer = keras.layers.deserialize(
    serialized_layer, custom_objects={"CustomLayer": CustomLayer}
)
```

**注册自定义对象**

Keras 会记录生成 config 的类。对上例，[tf.keras.layers.serialize](https://tensorflow.google.cn/api_docs/python/tf/keras/layers/serialize) 生成自定义 layer 的序列化形式：

```json
{'class_name': 'CustomLayer', 'config': {'a': 2} }
```

Keras 保存了所有内置 layer, model, optimizer 以及 metric 类的主列表，用于查找调用 `from_config` 所需的类。如果无法找到对应类，抛出 `Value Error: Unknown layer`。注册自定义类到主列表的方法有几种：

1. 在加载函数中设置 `custom_objects` 参数（参考上面的 "定义 config 方法"）。
2. `tf.keras.utils.custom_object_scope` 或 `tf.keras.utils.CustomObjectScope`
3. `tf.keras.utils.reegister_keras_serializable`

**自定义 layer 和函数示例**

```python
class CustomLayer(keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(CustomLayer, self).get_config()
        config.update({"units": self.units})
        return config


def custom_activation(x):
    return tf.nn.tanh(x) ** 2


# Make a model with the CustomLayer and custom_activation
inputs = keras.Input((32,))
x = CustomLayer(32)(inputs)
outputs = keras.layers.Activation(custom_activation)(x)
model = keras.Model(inputs, outputs)

# Retrieve the config
config = model.get_config()

# At loading time, register the custom objects with a `custom_object_scope`:
custom_objects = {"CustomLayer": CustomLayer, "custom_activation": custom_activation}
with keras.utils.custom_object_scope(custom_objects):
    new_model = keras.Model.from_config(config)
```

**内存中模型克隆**

可以使用 `tf.keras.models.clone_model()` 在内存中克隆模型。该操作等价于先获取 config，然后使用该 config 重新创建模型（因此它不保留编译信息和 layer 权重值）。

例如：

```python
with keras.utils.custom_object_scope(custom_objects):
    new_model = keras.models.clone_model(model)
```

## 仅保存和加载模型的权重

可以选择只保存和加载模型的权重，在以下情况很有用：

- 只需要使用模型进行推理：此时不需要重新训练，所以不需要编译信息或 optimizer 状态；
- 进行迁移学习，此时重用上一个模型的状态训练一个新模型，因此不需要上一个模型的编译信息。

### 内存中权重迁移 API

可以使用 `get_weights` 和 `set_weights` 在不同对象之间复制权重：

- `tf.keras.layers.Layer.get_weights()` 返回 numpy 数组 list；
- `tf.keras.layers.Layer.set_weights()` 将模型权重设置为参数 `weights` 的值。

示例如下。

**在内存中将权重从一个 layer 迁移到另一个 layer**

```python
def create_layer():
    layer = keras.layers.Dense(64, activation="relu", name="dense_2")
    layer.build((None, 784))
    return layer


layer_1 = create_layer()
layer_2 = create_layer()

# Copy weights from layer 1 to layer 2
layer_2.set_weights(layer_1.get_weights())
```

**在内存中将权重从一个 model 迁移到架构兼容的另一个 model**

```python
# Create a simple functional model
inputs = keras.Input(shape=(784,), name="digits")
x = keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = keras.layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = keras.layers.Dense(10, name="predictions")(x)
functional_model = keras.Model(inputs=inputs, outputs=outputs, name="3_layer_mlp")


# Define a subclassed model with the same architecture
class SubclassedModel(keras.Model):
    def __init__(self, output_dim, name=None):
        super(SubclassedModel, self).__init__(name=name)
        self.output_dim = output_dim
        self.dense_1 = keras.layers.Dense(64, activation="relu", name="dense_1")
        self.dense_2 = keras.layers.Dense(64, activation="relu", name="dense_2")
        self.dense_3 = keras.layers.Dense(output_dim, name="predictions")

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return x

    def get_config(self):
        return {"output_dim": self.output_dim, "name": self.name}


subclassed_model = SubclassedModel(10)
# Call the subclassed model once to create the weights.
subclassed_model(tf.ones((1, 784)))

# Copy weights from functional_model to subclassed_model.
subclassed_model.set_weights(functional_model.get_weights())

assert len(functional_model.weights) == len(subclassed_model.weights)
for a, b in zip(functional_model.weights, subclassed_model.weights):
    np.testing.assert_allclose(a.numpy(), b.numpy())
```

**stateless layer 的情况**

因为 stateless layer 不会改变权重的顺序，也不会改变权重的数量，所以即使包含额外或缺失 stateless layer，模型的架构也兼容。

```python
inputs = keras.Input(shape=(784,), name="digits")
x = keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = keras.layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = keras.layers.Dense(10, name="predictions")(x)
functional_model = keras.Model(inputs=inputs, outputs=outputs, name="3_layer_mlp")

inputs = keras.Input(shape=(784,), name="digits")
x = keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = keras.layers.Dense(64, activation="relu", name="dense_2")(x)

# Add a dropout layer, which does not contain any weights.
x = keras.layers.Dropout(0.5)(x)
outputs = keras.layers.Dense(10, name="predictions")(x)
functional_model_with_dropout = keras.Model(
    inputs=inputs, outputs=outputs, name="3_layer_mlp"
)

functional_model_with_dropout.set_weights(functional_model.get_weights())
```

### 权重到文件的保存和加载 API

调用 `model.save_weights` 可以将权重保存为以下格式：

- TF Checkpoint
- HDF5

`model.save_weights` 的默认格式为 TF checkpoint。指定保存格式的方法有两种：

1. `save_format` 参数：将其设置为 `save_format="tf"` 或 `save_format="h5"`；
2. `path` 参数：如果 path 以 `.h5` 或 `.hdf5` 结尾，则使用 HDF5 格式。其它后缀在不指定 `save_format` 参数时使用 TF checkpoint 格式。

每个 API 都尤其优缺点，详情如下。

### TF Checkpoint 格式

例如：

```python
# Runnable example
sequential_model = keras.Sequential(
    [
        keras.Input(shape=(784,), name="digits"),
        keras.layers.Dense(64, activation="relu", name="dense_1"),
        keras.layers.Dense(64, activation="relu", name="dense_2"),
        keras.layers.Dense(10, name="predictions"),
    ]
)
sequential_model.save_weights("ckpt")
load_status = sequential_model.load_weights("ckpt")

# `assert_consumed` can be used as validation that all variable values have been
# restored from the checkpoint. See `tf.train.Checkpoint.restore` for other
# methods in the Status object.
load_status.assert_consumed()
```

```txt
<tensorflow.python.checkpoint.checkpoint.CheckpointLoadStatus at 0x24b33fe1310>
```

TF Checkpoint 格式使用对象属性名称保存和恢复权重。例如，对 `tf.keras.layers.Dense` layer，包含两个权重：`dense.kernel` 和 `dense.bias`。当将 layer 保存为 `tf` 格式，生成的 checkpoint 包含 keys `"kernel"` 和 `"bias"` 及其相应的权重值。更多信息可参考 [TF Checkpoint 指南](https://tensorflow.google.cn/guide/checkpoint)。

需要注意的是，attribute/graph edge 是根据父对象中使用的名称命名的，而不是根据变量的名称命名。下例中 `CustomLayer`，变量 `CustomLayer.var` 是 `"var"` 作为 key 保存，而不是 `"var_a"`。

```python
class CustomLayer(keras.layers.Layer):
    def __init__(self, a):
        self.var = tf.Variable(a, name="var_a")


layer = CustomLayer(5)
layer_ckpt = tf.train.Checkpoint(layer=layer).save("custom_layer")

ckpt_reader = tf.train.load_checkpoint(layer_ckpt)

ckpt_reader.get_variable_to_dtype_map()
```

```txt
{'_CHECKPOINTABLE_OBJECT_GRAPH': tf.string,
 'layer/var/.ATTRIBUTES/VARIABLE_VALUE': tf.int32,
 'save_counter/.ATTRIBUTES/VARIABLE_VALUE': tf.int64}
```

**迁移学习示例**

只要两个模型具有相同的架构，它们就可以共享相同的 checkpoint。

例如：

```python
inputs = keras.Input(shape=(784,), name="digits")
x = keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = keras.layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = keras.layers.Dense(10, name="predictions")(x)
functional_model = keras.Model(inputs=inputs, outputs=outputs, name="3_layer_mlp")

# Extract a portion of the functional model defined in the Setup section.
# The following lines produce a new model that excludes the final output
# layer of the functional model.
pretrained = keras.Model(
    functional_model.inputs, functional_model.layers[-1].input, name="pretrained_model"
)
# Randomly assign "trained" weights.
for w in pretrained.weights:
    w.assign(tf.random.normal(w.shape))
pretrained.save_weights("pretrained_ckpt")
pretrained.summary()

# Assume this is a separate program where only 'pretrained_ckpt' exists.
# Create a new functional model with a different output dimension.
inputs = keras.Input(shape=(784,), name="digits")
x = keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = keras.layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = keras.layers.Dense(5, name="predictions")(x)
model = keras.Model(inputs=inputs, outputs=outputs, name="new_model")

# Load the weights from pretrained_ckpt into model.
model.load_weights("pretrained_ckpt")

# Check that all of the pretrained weights have been loaded.
for a, b in zip(pretrained.weights, model.weights):
    np.testing.assert_allclose(a.numpy(), b.numpy())

print("\n", "-" * 50)
model.summary()

# Example 2: Sequential model
# Recreate the pretrained model, and load the saved weights.
inputs = keras.Input(shape=(784,), name="digits")
x = keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = keras.layers.Dense(64, activation="relu", name="dense_2")(x)
pretrained_model = keras.Model(inputs=inputs, outputs=x, name="pretrained")

# Sequential example:
model = keras.Sequential([pretrained_model, keras.layers.Dense(5, name="predictions")])
model.summary()

pretrained_model.load_weights("pretrained_ckpt")

# Warning! Calling `model.load_weights('pretrained_ckpt')` won't throw an error,
# but will *not* work as expected. If you inspect the weights, you'll see that
# none of the weights will have loaded. `pretrained_model.load_weights()` is the
# correct method to call.
```

```txt
Model: "pretrained_model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 digits (InputLayer)         [(None, 784)]             0         
                                                                 
 dense_1 (Dense)             (None, 64)                50240     
                                                                 
 dense_2 (Dense)             (None, 64)                4160      
                                                                 
=================================================================
Total params: 54,400
Trainable params: 54,400
Non-trainable params: 0
_________________________________________________________________

 --------------------------------------------------
Model: "new_model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 digits (InputLayer)         [(None, 784)]             0         
                                                                 
 dense_1 (Dense)             (None, 64)                50240     
                                                                 
 dense_2 (Dense)             (None, 64)                4160      
                                                                 
 predictions (Dense)         (None, 5)                 325       
                                                                 
=================================================================
Total params: 54,725
Trainable params: 54,725
Non-trainable params: 0
_________________________________________________________________
Model: "sequential_3"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 pretrained (Functional)     (None, 64)                54400     
                                                                 
 predictions (Dense)         (None, 5)                 325       
                                                                 
=================================================================
Total params: 54,725
Trainable params: 54,725
Non-trainable params: 0
_________________________________________________________________

<tensorflow.python.checkpoint.checkpoint.CheckpointLoadStatus at 0x24b49b74850>
```

通常建议使用相同的 API 构建模型。如果在 Sequential 和 Functional，或 Functional 和 Subclassed 等之间切换，则始终重新构建预训练模型，然后加载预训练权重。

下一个问题是，如果模型架构完全不同，如何保存权重、然后在不同的模型中加载？解决方案是使用 `tf.train.Checkpoint` 保存和恢复特定 layers/variables。

例如：

```python
# Create a subclassed model that essentially uses functional_model's first
# and last layers.
# First, save the weights of functional_model's first and last dense layers.
first_dense = functional_model.layers[1]
last_dense = functional_model.layers[-1]
ckpt_path = tf.train.Checkpoint(
    dense=first_dense, kernel=last_dense.kernel, bias=last_dense.bias
).save("ckpt")


# Define the subclassed model.
class ContrivedModel(keras.Model):
    def __init__(self):
        super(ContrivedModel, self).__init__()
        self.first_dense = keras.layers.Dense(64)
        self.kernel = self.add_variable("kernel", shape=(64, 10))
        self.bias = self.add_variable("bias", shape=(10,))

    def call(self, inputs):
        x = self.first_dense(inputs)
        return tf.matmul(x, self.kernel) + self.bias


model = ContrivedModel()
# Call model on inputs to create the variables of the dense layer.
_ = model(tf.ones((1, 784)))

# Create a Checkpoint with the same structure as before, and load the weights.
tf.train.Checkpoint(
    dense=model.first_dense, kernel=model.kernel, bias=model.bias
).restore(ckpt_path).assert_consumed()
```

```txt
C:\Users\happy\AppData\Local\Temp\ipykernel_1168\1215903662.py:16: UserWarning: `layer.add_variable` is deprecated and will be removed in a future version. Please use the `layer.add_weight()` method instead.
  self.kernel = self.add_variable("kernel", shape=(64, 10))
C:\Users\happy\AppData\Local\Temp\ipykernel_1168\1215903662.py:17: UserWarning: `layer.add_variable` is deprecated and will be removed in a future version. Please use the `layer.add_weight()` method instead.
  self.bias = self.add_variable("bias", shape=(10,))
Out[20]:
<tensorflow.python.checkpoint.checkpoint.CheckpointLoadStatus at 0x24b28a24550>
```

### HDF5 格式

HDF5 格式按 layer 名称分组保存权重。这些权重将 trainable 权重和 non-trainable 权重串联起来，顺序和 `layer.weights` 相同。因此，如果模型与保存的 checkpoint 具有相同的 layers 和 trainable 状态，则可以使用 hdf5 checkpoint。

例如：

```python
# Runnable example
sequential_model = keras.Sequential(
    [
        keras.Input(shape=(784,), name="digits"),
        keras.layers.Dense(64, activation="relu", name="dense_1"),
        keras.layers.Dense(64, activation="relu", name="dense_2"),
        keras.layers.Dense(10, name="predictions"),
    ]
)
sequential_model.save_weights("weights.h5")
sequential_model.load_weights("weights.h5")
```

注意，当模型包含嵌套 layer 时，更改 `layer.trainable` 可能产生不同的 `layer.weights` 顺序。

```python
class NestedDenseLayer(keras.layers.Layer):
    def __init__(self, units, name=None):
        super(NestedDenseLayer, self).__init__(name=name)
        self.dense_1 = keras.layers.Dense(units, name="dense_1")
        self.dense_2 = keras.layers.Dense(units, name="dense_2")

    def call(self, inputs):
        return self.dense_2(self.dense_1(inputs))


nested_model = keras.Sequential([keras.Input((784,)), NestedDenseLayer(10, "nested")])
variable_names = [v.name for v in nested_model.weights]
print("variables: {}".format(variable_names))

print("\nChanging trainable status of one of the nested layers...")
nested_model.get_layer("nested").dense_1.trainable = False

variable_names_2 = [v.name for v in nested_model.weights]
print("\nvariables: {}".format(variable_names_2))
print("variable ordering changed:", variable_names != variable_names_2)
```

```txt
variables: ['nested/dense_1/kernel:0', 'nested/dense_1/bias:0', 'nested/dense_2/kernel:0', 'nested/dense_2/bias:0']

Changing trainable status of one of the nested layers...

variables: ['nested/dense_2/kernel:0', 'nested/dense_2/bias:0', 'nested/dense_1/kernel:0', 'nested/dense_1/bias:0']
variable ordering changed: True
```

**迁移学习示例**

从 HDF5 加载预训练权重时，建议将权重加载到原始的 checkpoint 模型中，然后将所需的 weights/layeers 提取到新的模型。

例如：

```python
def create_functional_model():
    inputs = keras.Input(shape=(784,), name="digits")
    x = keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = keras.layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = keras.layers.Dense(10, name="predictions")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="3_layer_mlp")


functional_model = create_functional_model()
functional_model.save_weights("pretrained_weights.h5")

# In a separate program:
pretrained_model = create_functional_model()
pretrained_model.load_weights("pretrained_weights.h5")

# Create a new model by extracting layers from the original model:
extracted_layers = pretrained_model.layers[:-1]
extracted_layers.append(keras.layers.Dense(5, name="dense_3"))
model = keras.Sequential(extracted_layers)
model.summary()
```

```txt
Model: "sequential_6"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_1 (Dense)             (None, 64)                50240     
                                                                 
 dense_2 (Dense)             (None, 64)                4160      
                                                                 
 dense_3 (Dense)             (None, 5)                 325       
                                                                 
=================================================================
Total params: 54,725
Trainable params: 54,725
Non-trainable params: 0
_________________________________________________________________
```

## 参考

- https://www.tensorflow.org/guide/keras/save_and_serialize
- https://tensorflow.google.cn/guide/keras/save_and_serialize
