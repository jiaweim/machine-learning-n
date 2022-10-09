# Model

- [Model](#model)
  - [简介](#简介)
  - [参数](#参数)
  - [属性](#属性)
  - [方法](#方法)
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
