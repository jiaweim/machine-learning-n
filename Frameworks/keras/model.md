# Model

- [Model](#model)
  - [keras.Model](#kerasmodel)
    - [创建模型](#创建模型)
    - [属性](#属性)
    - [summary](#summary)
  - [keras.Sequential](#kerassequential)
  - [参考](#参考)

2021-06-04, 09:32
***

## keras.Model

`Model` 表示一个模型，可以理解为具有训练和预测功能的分组 layers。

```python
tf.keras.Model(
    *args, **kwargs
)
```

| **参数** | **说明** |
| --- | --- |
| inputs | 模型输入， `keras.Input` 对象或其列表 |
| outputs | 模型输出 |
| name | 模型名称 |

### 创建模型

实例化 `Model` 的方法有两种：

1. 使用函数 API

从 `Input` 开始，通过连接 layer 指定模型的前向传播，最后从输入和输出创建模型：

```python
import tensorflow as tf

inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

2. 扩展 `Model` 类

在 `__init__` 中定义 layers，在 `call` 方法中定义前向传播。例如：

```python
import tensorflow as tf

class MyModel(tf.keras.Model):

  def __init__(self):
    super(MyModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

  def call(self, inputs):
    x = self.dense1(inputs)
    return self.dense2(x)

model = MyModel()
```

扩展 `Model` 类可以在 `call` 中包含一个可选的 `training` 参数，用于指定在训练和预测时的不同行为：

```python
import tensorflow as tf

class MyModel(tf.keras.Model):

  def __init__(self):
    super(MyModel, self).__init__()
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

创建模型后，可以使用 `model.compile()` 配置损失函数（loss）和测量指标（metrices），使用 `model.fit()` 训练模型，使用 `model.predict()` 执行预测功能。

### 属性

- `distribute_strategy`

使用的 `tf.distribute.Strategy` 。

- `layers`

该模型包含的 layers。

- `metrics_names`

输出标签。
`metrics_names` 只有在 `keras.Model` 训练之后才可用。例如：

```python
inputs = tf.keras.layers.Input(shape=(3,))
outputs = tf.keras.layers.Dense(2)(inputs)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
model.metrics_names
# []
```

```python
x = np.random.random((2, 3))
y = np.random.randint(0, 2, (2, 2))
model.fit(x, y)
model.metrics_names
# ['loss', 'mae']
```

```python
inputs = tf.keras.layers.Input(shape=(3,))
d = tf.keras.layers.Dense(2, name='out')
output_1 = d(inputs)
output_2 = d(inputs)
model = tf.keras.models.Model(
   inputs=inputs, outputs=[output_1, output_2])
model.compile(optimizer="Adam", loss="mse", metrics=["mae", "acc"])
model.fit(x, (y, y))
model.metrics_names
# ['loss', 'out_loss', 'out_1_loss', 'out_mae', 'out_acc', 'out_1_mae', 'out_1_acc']
```

### summary

```python
summary(
    line_length=None, positions=None, print_fn=None
)
```

输出模型的描述信息。

## keras.Sequential

`Sequential` 以线性的方式组织 layers 创建模型。

```python
tf.keras.Sequential(
    layers=None, name=None
)
```

## 参考

- https://keras.io/api/models/
