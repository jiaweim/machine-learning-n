# 自定义 Layer 和 Model

2022-03-08, 23:39
***

## 设置

```python
import tensorflow as tf
from tensorflow import keras
```

## Layer 类：（权重和计算的组合）

Keras 的核心抽象之一是 `Layer` 类。layer 类封装状态（layer 的权重）以及输入到输出的转换（"call" 方法，向前传播）。

下面是一个密集连接层。它状态为：变量 `w` 和 `b`：

```python
class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```

可以像使用 Python 函数一样使用 layer，输出为张量：

```python
x = tf.ones((2, 2))
linear_layer = Linear(4, 2)
y = linear_layer(x)
print(y)
```

```sh
tf.Tensor(
[[ 0.00962844 -0.01307489 -0.1452128   0.0538918 ]
 [ 0.00962844 -0.01307489 -0.1452128   0.0538918 ]], shape=(2, 4), dtype=float32)
```

注意，将 `w` 和 `b` 设置为 layer 属性后，layer 会自动跟踪权重：

```python
assert linear_layer.weights == [linear_layer.w, linear_layer.b]
```

也可以使用快捷方式 `add_weight()` 增加权重：

```python
class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        self.w = self.add_weight(
            shape=(input_dim, units), initializer="random_normal", trainable=True
        )
        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


x = tf.ones((2, 2))
linear_layer = Linear(4, 2)
y = linear_layer(x)
print(y)
```

```sh
tf.Tensor(
[[ 0.05790994  0.060931   -0.0402256  -0.09450993]
 [ 0.05790994  0.060931   -0.0402256  -0.09450993]], shape=(2, 4), dtype=float32)
```

## non-trainable weights

除了可训练权重，layer 可以包含不可训练权重。在训练的时候，反向传播不考虑这些权重值。

添加 non-trainable weight 方法：

```python
class ComputeSum(keras.layers.Layer):
    def __init__(self, input_dim):
        super(ComputeSum, self).__init__()
        self.total = tf.Variable(initial_value=tf.zeros((input_dim,)), trainable=False)

    def call(self, inputs):
        self.total.assign_add(tf.reduce_sum(inputs, axis=0))
        return self.total


x = tf.ones((2, 2))
my_sum = ComputeSum(2)
y = my_sum(x)
print(y.numpy())
y = my_sum(x)
print(y.numpy())
```

```sh
[2. 2.]
[4. 4.]
```

`total` 是 `layer.weights` 的一部分，但是属于 non-trainable weight:

```python
print("weights:", len(my_sum.weights))
print("non-trainable weights:", len(my_sum.non_trainable_weights))

# It's not included in the trainable weights:
print("trainable_weights:", my_sum.trainable_weights)
```

```sh
weights: 1
non-trainable weights: 1
trainable_weights: []
```

## 最佳实践：将 weight 的创建推迟到输入 shape 已知

上面的 `Linear` 层在 `__init__()` 中根据参数 `input_dim` 计算权重 `w` 和 `b` 的 shape:

```python
class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        self.w = self.add_weight(
            shape=(input_dim, units), initializer="random_normal", trainable=True
        )
        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```

但是很多时候，我们可能事先不知道输入的大小，因此希望在实例化该层之后，在知道 shape 后再 lazily 创建 weights。

在 Keras API 中，我们建议在 layer 的 `build(self, inputs_shape)` 方法中创建 weights。如下：

```python
class Linear(keras.layers.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
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
```

layer 的 `__call__()` 方法在第一次调用时会自动运行 build 方法。lazy 初始化的 layer，使用更容易：

```python
# 实例化时，我们不知道输入 shape
linear_layer = Linear(32)

# 第一次调用 layer 时动态创建 layer 的权重
y = linear_layer(x)
```

如上所示，单独实现 `build()` 可以很好地将权重的创建与使用分开。然而，对于一些高级自定义 layer，将状态创建和计算分开几乎不可能。layer 的创建者依然可以将权重的创建推迟到第一次调用 `__call__()`
，但是要注意以后的调用使用相同的权重。另外，`__call__()` 第一次执行很可能在 `tf.function` 中，因此 `__call__()` 中创建任何变量都应该放在 `tf.init_scope` 中。

## Layer 可递归组合

创建将一个 layer 实例作为另一个 layer 的属性，则外层 layer 会自动跟踪内层创建的权重。

建议在 `__init__()` 中创建这样的 sublayers，权重则由第一次调用 `__call__()` 时触发构建。

```python
class MLPBlock(keras.layers.Layer):
    def __init__(self):
        super(MLPBlock, self).__init__()
        self.linear_1 = Linear(32)
        self.linear_2 = Linear(32)
        self.linear_3 = Linear(1)

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        return self.linear_3(x)


mlp = MLPBlock()
y = mlp(tf.ones(shape=(3, 64)))  # The first call to the `mlp` will create the weights
print("weights:", len(mlp.weights))
print("trainable weights:", len(mlp.trainable_weights))
```

```sh
weights: 6
trainable weights: 6
```

## add_loss()

在 `call()` 方法中可以创建在训练循环时要使用的损失张量，通过调用 `self.add_loss(value)` 实现：

```python
# A layer that creates an activity regularization loss
class ActivityRegularizationLayer(keras.layers.Layer):
    def __init__(self, rate=1e-2):
        super(ActivityRegularizationLayer, self).__init__()
        self.rate = rate

    def call(self, inputs):
        self.add_loss(self.rate * tf.reduce_sum(inputs))
        return inputs
```

这些 losses (包含内部 layer 创建的 loss)可以通过 `layer.losses` 查询。

## 参考

- https://www.tensorflow.org/guide/keras/custom_layers_and_models
