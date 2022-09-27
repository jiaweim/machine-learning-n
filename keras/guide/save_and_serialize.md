# Keras 模型的保存和加载

- [Keras 模型的保存和加载](#keras-模型的保存和加载)
  - [简介](#简介)
  - [如何保存和加载模型](#如何保存和加载模型)
  - [设置](#设置)
  - [整个模型的保存和加载](#整个模型的保存和加载)
    - [SavedModel 格式](#savedmodel-格式)
  - [参考](#参考)

***

## 简介

Keras 模型由多个组件组成：

- 模型架构，指定模型包含哪些 layer，以及这些 layer 如何连接的；
- 权重值（模型状态，state of the model）；
- 优化器（optimizer），通过 compile 定义；
- losses 和 metrics，通过 compile 或调用 `add_loss()`, `add_metric()` 定义。

Keras 支持保存所有这些组件，也可以选择性地保存一部分：

- 将所有组件保存为单个 TF SavedModel 归档格式（或老的 Keras H5 格式），这是标准做法；
- 只保存模型架构，通常保存为 JSON 文件；
- 只保存权重，一般在训练模型时使用。

下面依次查看这些选项，以及如何使用它们。

## 如何保存和加载模型

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

## 整个模型的保存和加载

可以将整个模型保存为单个构建（artifact）。包括：

- 模型架构
- 模型权重
- 模型编译信息
- optimizer 及其状态（便于从中断的地方重新开始训练）

**APIs**

- `model.save()` 或 [tf.keras.models.save_model()](https://www.tensorflow.org/api_docs/python/tf/keras/models/save_model)
- [tf.keras.models.load_model()](https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model)

保存整个模型有两种格式可选：TF SavedModel 格式，以及老的 Keras **H5** 格式。SavedModel 是推荐格式，是使用 `model.save()` 的默认选项。

切换到 H5 格式的方法：

- `save()` 中设置 `save_format='h5'`
- `save()` 中传入的文件名以 `.h5` 或 `.keras` 结尾

### SavedModel 格式

SavedModel 是一种更全面的格式，能保存模型架构、权重以及调用函数的 TF subgraph。使得 Keras 能够恢复内置 layer 和自定义对象。例如：

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
4/4 [==============================] - 1s 2ms/step - loss: 0.2814
INFO:tensorflow:Assets written to: my_model\assets
4/4 [==============================] - 0s 1ms/step
4/4 [==============================] - 0s 1ms/step
4/4 [==============================] - 0s 2ms/step - loss: 0.2716
<keras.callbacks.History at 0x19edd0a7eb0>
```

**SavedModel 包含内容**

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

在没有 model/layer 配置的情况下，调用函数可用来创建一个类似原始模型的模型，该模型可以训练、评估和推理。

然后，在自定义模型或 layer 时，最好定义 `get_config` 和 `from_config` 方法。以便在需要时更容易更新计算。

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




## 参考

- https://www.tensorflow.org/guide/keras/save_and_serialize
