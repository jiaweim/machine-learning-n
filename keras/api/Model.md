# Model

- [Model](#model)
  - [简介](#简介)
  - [参数](#参数)
  - [属性](#属性)
  - [方法](#方法)
    - [compile](#compile)
    - [get_layer](#get_layer)
    - [save](#save)
    - [to_yaml](#to_yaml)
  - [参考](#参考)

2022-03-09, 21:59
***

## 简介

```python
tf.keras.Model(
    *args, **kwargs
)
```

`Model` 类表示模型，将多个 layer 组合在一起，并包含训练和推断功能。

实例化 `Model` 的方法有两种：

**1. 使用函数 API**

从 `Input` 开始，通过 layer 调用指定模型的正向传播，最后从输入和输出创建模型：

```python
import tensorflow as tf

inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

> ⚡：输入只支持张量类型的 dict, list 或 tuple。目前不支持嵌套。

也可以使用中间张量创建新的函数 API 模型，这样就能快速提取模型的子模块。例如：

```python
inputs = keras.Input(shape=(None, None, 3))
processed = keras.layers.RandomCrop(width=32, height=32)(inputs)
conv = keras.layers.Conv2D(filters=2, kernel_size=3)(processed)
pooling = keras.layers.GlobalAveragePooling2D()(conv)
feature = keras.layers.Dense(10)(pooling)

full_model = keras.Model(inputs, feature)
backbone = keras.Model(processed, conv)
activations = keras.Model(conv, feature)
```

请注意，`backbone` 和 `activations` 模型不是从 `keras.Input` 对象创建，而是使用源自 `keras.Input` 对象的张量。在底层，这些模型共享 layers 和 weights，这样用户就可以训练 `full_model`，并使用 `backbone` 和 `activations` 提取特征。模型的输入和输出也可以是嵌套张量，创建的模型为标准函数 API 模型，支持所有现有 API。

**2. 扩展 `Model` 类**

此时需要在 `__init__()` 中定义 layers，在 `call()` 中实现前向传播。

```python
import tensorflow as tf

class MyModel(tf.keras.Model):

  def __init__(self):
    super().__init__()
    self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

  def call(self, inputs):
    x = self.dense1(inputs)
    return self.dense2(x)

model = MyModel()
```

扩展 `Model` 类时，可以选择在 `call()` 方法中包含 `training` 参数，用于指定在训练和推理时的不同行为：

```python
import tensorflow as tf

class MyModel(tf.keras.Model):

  def __init__(self):
    super().__init__()
    self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
    self.dropout = tf.keras.layers.Dropout(0.5)

  def call(self, inputs, training=False):
    x = self.dense1(inputs)
    if training:
      x = self.dropout(x, training=training)
    return self.dense2(x)

model = MyModel()
```

创建模型后，可使用 `model.compile()` 设置模型的损失函数和评价指标，使用 `model.fit()` 训练模型，使用 `model.predict()` 做预测。

## 参数

|参数|说明|
|---|---|
|inputs|模型输入，`keras.Input` 对象或其列表|
|outputs|模型输出|
|name|String, 模型名称|

## 属性

- distribute_strategy

## 方法

### compile

Last updated: 2022-10-09, 17:27

```python
compile(
    optimizer='rmsprop',
    loss=None,
    metrics=None,
    loss_weights=None,
    weighted_metrics=None,
    run_eagerly=None,
    steps_per_execution=None,
    jit_compile=None,
    **kwargs
)
```

配置模型的训练参数。

例如：

```python
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.FalseNegatives()])
```

|参数|说明|
|---|---|
|optimizer|String (optimizer 名称) 或 optimizer 实例。参考 [tf.keras.optimizers](https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers)|
|loss|损失函数。可以是 string (损失函数名称) 或 [tf.keras.losses.Loss](https://tensorflow.google.cn/api_docs/python/tf/keras/losses/Loss) 实例。参考 [tf.keras.losses](https://tensorflow.google.cn/api_docs/python/tf/keras/losses)。损失函数是签名为 `loss = fn(y_true, y_pred)` 的可调用对象，其中 `y_true` 是真实值，`y_pred` 是模型推理值。`y_true` 的 shape 应为 `(batch_size, d0, ...dN)`（稀疏损失函数除外，如稀疏分类交叉熵，需要 shape 为 `(batch_size, d0, ...dN-1)` 的整数数组）。`y_pred` 的 shape 应为 `(batch_size, d0, .. dN)`。损失函数应返回 float 张量。如果使用自定义 `Loss` 并且 reeduction 设置为 `None`，则返回 shape 为 `(batch_size, d0, .. dN-1)`，即每个样本或每个时间步一个损失值，否则为标量。如果模型有多个输出，可以传入 loss dict 或 list 为每个输出使用不同的损失函数，此时模型最小化的 loss 值为每个单独 loss 的加和，除非使用 `loss_weights`|
|metrics|训练和测试期间评估模型的 metric 列表。每个 metric 可以是 string（内置函数名称）、函数或 `tf.keras.metrics.Metric` 实例。通常使用 `metrics=['accuracy']`。函数是签名为 `result = fn(y_true, y_pred)` 的任意可调用函数。对多输出模型，可以用 dict 为不同输出设置不同 metrics，例如 `metrics={'output_a':'accuracy', 'output_b':['accuracy', 'mse']}`。也可以用 list 为每个输出指定单个 metric 或 metric 列表，例如 `metrics=[['accuracy'], ['accuracy', 'mse']]` 或 `metrics=['accuracy', ['accuracy', 'mse']]`。当传入字符串 'accuracy' 或 'acc'，则根据使用的损失函数和模型输出 shape 转换为 [tf.keras.metrics.BinaryAccuracy](https://tensorflow.google.cn/api_docs/python/tf/keras/metrics/BinaryAccuracy), [tf.keras.metrics.CategoricalAccuracy](https://tensorflow.google.cn/api_docs/python/tf/keras/metrics/CategoricalAccuracy) 或 [tf.keras.metrics.SparseCategoricalAccuracy](https://tensorflow.google.cn/api_docs/python/tf/keras/metrics/SparseCategoricalAccuracy) 三者之一。对字符串 'crossentropy' 和 'ce' 也进行类似转换。此处传入的 metrics 不使用样本加权进行计算，如果需要样本加权，则应该使用 `weighted_metrics` 参数|
|loss_weights|（可选）Python float 的 list 或 dict，指定模型不同输出对 loss 的贡献。模型最小化的 loss 值为所有输出 loss 的加权和，加权值为 `loss_weights`。若为 list，则应和模型的输出一一对应；若为 dict，则将名称（string）映射为系数（标量）|
|weighted_metrics|训练和测试期间根据 `sample_weight` 或 `class_weight` 计算的加权 metric list|
|run_eagerly|Bool. 默认 `False`。`True` 表示不将模型的逻辑包装在 [tf.function](https://tensorflow.google.cn/api_docs/python/tf/function)。建议将其保留为 `None`，除非模型不能在 `tf.function` 中运行。使用 [tf.distribute.experimental.ParameterServerStrategy](https://tensorflow.google.cn/api_docs/python/tf/distribute/experimental/ParameterServerStrategy) 时不支持 `run_eagerly=True`|
|steps_per_execution|Int. 默认 1。每次 `tf.function` 调用运行的 batch 数。在单个 `tf.function` 中运行多个 batch 在 TPU 中或小型模型中可以极大提高性能，但 Python 开销更大。每次执行最多运行一个完整的 epoch。如果传入的数字大于 epoch 大小，在执行时将截断为 epoch 的大小。注意，如果 `steps_per_execution` 设置为  N，则 `Callback.on_batch_begin` 和 `Callback.on_batch_end` 方法会每 N 个 batch 调用调用一次（即在每次 `tf.function` 执行前/后调用）|
|jit_compile|`True` 表示使用 XLA 编译模型的训练步骤。[XLA](https://tensorflow.google.cn/xla) 是一种用于机器学习的优化编译器，默认不启用。设置 `run_eagerly=True` 时无法启用。注意 `jit_compile=True` 不一定适用于所有模型。对 XLA 支持的操作，可参考 [XLA 文档](https://tensorflow.google.cn/xla)|
|**kwargs|仅用于向后兼容的参数|

### get_layer

Last updated: 2022-08-09, 14:01

```python
get_layer(
    name=None, index=None
)
```

|参数|说明|
|---|---|
|name| layer 名称|
|index| layer 索引|

**返回**： `layer` 实例。

根据名称 `name` 或索引 `index` 查找 layer。

如果同时提供 `name` 和 `index`，则 `index` 优先。layer 基于水平图遍历（bottom-up）进行索引。

### save

Last updated: 2022-10-09, 14:57

```python
save(
    filepath,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None,
    save_traces=True
)
```

将模型保存为 TF SavedModel 或单个 HDF5 文件。

具体可参考 [tf.keras.models.save_model](https://tensorflow.google.cn/api_docs/python/tf/keras/models/save_model) 或 [Keras 模型的保存和加载指南](https://tensorflow.google.cn/guide/keras/save_and_serialize)。

|参数|说明|
|---|---|
|filepath|String, PathLike, 保存模型的 SavedModel 或 H5 文件路径|
|overwrite|是否静静地覆盖目标位置的现有文件，还是给用户提示，默认 True|
|include_optimizer|是否保存 optimizer 的状态，默认 True|
|save_format|'tf' 或 'h5'，将模型保存为 TF SavedModel 还是 HDF5 格式。TF 2.x 中默认为 'tf'，TF 1.x 中默认为 'h5'|
|signatures|与 SavedModel 一同保存的签名。只适用于 "tf" 格式。详情请参考 [tf.saved_model.save](https://tensorflow.google.cn/api_docs/python/tf/saved_model/save)|
|options|(仅适用于 SavedModel 格式) [tf.saved_model.SaveOptions](https://tensorflow.google.cn/api_docs/python/tf/saved_model/SaveOptions) 对象，用于指定保存为 SavedModel 的选项|
|save_traces|(仅适用于 SavedModel 格式) 启用后，SavedModel 将保存每个 layer 的函数 traces。可以禁用该选项，这样就只保存每层的 config。**默认** True。禁用此选项可以减少序列化时间和文件大小，但要求所有自定义 layer/model 实现 `get_config()` 方法|

示例：

```python
from keras.models import load_model

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')
```

### to_yaml

```python
to_yaml(
    **kwargs
)
```

返回包含网络配置的 yaml 字符串。

> **!NOTE**：从 TF 2.6 开始不支持该方法，调用抛出 `RuntimeError`。

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/Model
- https://tensorflow.google.cn/api_docs/python/tf/keras/Model
