# 使用子类 API 创建 Layer 和 Model

- [使用子类 API 创建 Layer 和 Model](#使用子类-api-创建-layer-和-model)
  - [1. 设置](#1-设置)
  - [2. Layer 类：权重和计算的组合](#2-layer-类权重和计算的组合)
  - [3. 不可训练权重](#3-不可训练权重)
  - [4. 将 weight 的创建推迟到输入 shape 已知](#4-将-weight-的创建推迟到输入-shape-已知)
  - [5. Layer 可递归组合](#5-layer-可递归组合)
  - [6. add_loss](#6-add_loss)
  - [7. add_metric](#7-add_metric)
  - [8. 启用 layer 序列化](#8-启用-layer-序列化)
  - [9. call() 方法的 training 参数](#9-call-方法的-training-参数)
  - [10. call() 方法的 mask 参数](#10-call-方法的-mask-参数)
  - [11. Model 类](#11-model-类)
  - [12. 完整示例](#12-完整示例)
  - [13. 函数 API](#13-函数-api)
  - [14. 参考](#14-参考)

Last updated: 2022-07-06, 16:11
@author Jiawei Mao
****

## 1. 设置

```python
import tensorflow as tf
from tensorflow import keras
```

## 2. Layer 类：权重和计算的组合

`Layer` 类是 Keras 的核心抽象之一，它封装了状态（layer 权重）和输入到输出的转换（"call" 方法，layer 的前向传播）。

下面是一个全连接层，变量 `w` 和 `b` 是其状态：

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
            initial_value=b_init(shape=(units,), dtype="float32"),
            trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```

可以像使用 Python 函数一样调用 layer：

```python
x = tf.ones((2, 2))
linear_layer = Linear(4, 2)
y = linear_layer(x)
print(y)
```

```bash
tf.Tensor(
[[ 0.01460802 -0.02662525  0.07070637 -0.01873659]
 [ 0.01460802 -0.02662525  0.07070637 -0.01873659]], shape=(2, 4), dtype=float32)
```

> [!NOTE]
> 将 `w` 和 `b` 设置为 layer 属性后，layer 会自动跟踪权重。

```python
assert linear_layer.weights == [linear_layer.w, linear_layer.b]
```

也可以使用快捷方式 `add_weight()` 添加权重：

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

```bash
tf.Tensor(
[[-0.02559814  0.07031661 -0.07307922 -0.00163199]
 [-0.02559814  0.07031661 -0.07307922 -0.00163199]], shape=(2, 4), dtype=float32)
```

对比前面的定义，可以发现，`add_weight` 和定义 `tf.Variable` 代码形式基本一致，可以看作语法糖。

## 3. 不可训练权重

除了可训练权重，layer 可以包含不可训练权重。在训练时，反向传播不更新不可训练权重的值。

添加不可训练权重的方法：

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

```bash
[2. 2.]
[4. 4.]
```

`total` 是 `layer.weights` 的一部分，但是属于不可训练权重:

```python
print("weights:", len(my_sum.weights))
print("non-trainable weights:", len(my_sum.non_trainable_weights))

print("trainable_weights:", my_sum.trainable_weights)
```

```sh
weights: 1
non-trainable weights: 1
trainable_weights: []
```

## 4. 将 weight 的创建推迟到输入 shape 已知

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

但是很多时候，事先不知道输入的大小，因此希望在知道 shape 后再 lazily 创建 weights。

在 Keras API 中，建议在 layer 的 `build(self, inputs_shape)` 方法中创建 weights。如下：

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

layer 的 `__call__()` 方法在第一次调用时会自动运行 `build` 方法。lazy 初始化的 layer，使用更容易：

```python
# 实例化时，不知道输入 shape
linear_layer = Linear(32)

# 第一次调用 layer 时动态创建 layer 的权重
y = linear_layer(x)
```

如上所示，单独实现 `build()` 可以很好地将权重的创建与使用分开。然而，对一些高级自定义 layer，将状态创建和计算分开几乎不可能。layer 创建者依然可以将权重的创建推迟到第一次调用 `__call__()`
，但是要注意以后的调用使用相同的权重。另外，`__call__()` 第一次执行很可能在 `tf.function` 中，因此 `__call__()` 中创建任何变量都应该放在 `tf.init_scope` 中。

## 5. Layer 可递归组合

如果将一个 layer 实例作为另一个 layer 的属性，则外层 layer 会自动跟踪内层 layer 的权重。

建议在 `__init__()` 中创建 sublayers，权重则由第一次调用 `__call__()` 时触发构建。

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
y = mlp(tf.ones(shape=(3, 64)))  # 第一次调用 `mlp` 时触发创建 weights
print("weights:", len(mlp.weights))
print("trainable weights:", len(mlp.trainable_weights))
```

```bash
weights: 6
trainable weights: 6
```

## 6. add_loss

在 `call()` 方法中可以创建在训练循环时要使用的损失张量，通过调用 `self.add_loss(value)` 实现：

```python
# 创建输出正则化损失的 layer
class ActivityRegularizationLayer(keras.layers.Layer):
    def __init__(self, rate=1e-2):
        super(ActivityRegularizationLayer, self).__init__()
        self.rate = rate

    def call(self, inputs):
        self.add_loss(self.rate * tf.reduce_sum(inputs))
        return inputs
```

这些 losses (包括内部 layer 创建的 loss)可以通过 `layer.losses` 查询。该属性在每次调用顶层 layer 的 `__call__()` 方法时重置，因此 `layer.losses` 总是包含上次前向传播的损失值。

```python
class OuterLayer(keras.layers.Layer):
    def __init__(self):
        super(OuterLayer, self).__init__()
        self.activity_reg = ActivityRegularizationLayer(1e-2)

    def call(self, inputs):
        return self.activity_reg(inputs)


layer = OuterLayer()
assert len(layer.losses) == 0  # 由于还没调用 layer，此时没有损失值

_ = layer(tf.zeros(1, 1))
assert len(layer.losses) == 1  # 创建一个损失值

# `layer.losses` 在 `__call__` 开头重置
_ = layer(tf.zeros(1, 1))
assert len(layer.losses) == 1  # This is the loss created during the call above
```

此外，`loss` 属性还包含内层权重正则化损失：

```python
class OuterLayerWithKernelRegularizer(keras.layers.Layer):
    def __init__(self):
        super(OuterLayerWithKernelRegularizer, self).__init__()
        self.dense = keras.layers.Dense(
            32, kernel_regularizer=tf.keras.regularizers.l2(1e-3)
        )

    def call(self, inputs):
        return self.dense(inputs)


layer = OuterLayerWithKernelRegularizer()
_ = layer(tf.zeros((1, 1)))

# 等于 `1e-3 * sum(layer.dense.kernel ** 2)`,
# created by the `kernel_regularizer` above.
print(layer.losses)
```

```txt
[<tf.Tensor: shape=(), dtype=float32, numpy=0.0016654292>]
```

在编写训练循环时，应该考虑这些损失，例如：

```python
# 实例化优化器
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 迭代数据集的批次
for x_batch_train, y_batch_train in train_dataset:
    with tf.GradientTape() as tape:
        logits = layer(x_batch_train)  # 当前批次数据的 Logits
        # 当前批次的损失值
        loss_value = loss_fn(y_batch_train, logits)
        # 加上前向传播的其它损失
        loss_value += sum(model.losses)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
```

这些损失可以与 `fit()` 无缝配合，即如果有这些损失，它们会自动加到主损失：

```python
import numpy as np

inputs = keras.Input(shape=(3,))
outputs = ActivityRegularizationLayer()(inputs)
model = keras.Model(inputs, outputs)

# 如果 `compile` 包含损失，则加入正则化损失
model.compile(optimizer="adam", loss="mse")
model.fit(np.random.random((2, 3)), np.random.random((2, 3)))

# 在 `compile` 中也可以不设置损失
# 因此在前向传播中使用 `add_loss` 添加了损失，即已有需要需最小化的损失
model.compile(optimizer="adam")
model.fit(np.random.random((2, 3)), np.random.random((2, 3)))
```

```txt
1/1 [==============================] - 0s 164ms/step - loss: 0.1643
1/1 [==============================] - 0s 45ms/step - loss: 0.0303
<keras.callbacks.History at 0x18f8f9eb430>
```

## 7. add_metric

与 `add_loss()` 类似，layer 还有一个 `add_metric()` 方法，可用于追踪训练过程中指标的移动平均值。

例如，考虑下面的逻辑端点层，它以预测值和目标值为输入，通过 `add_loss()` 跟踪计算的损失，并通过 `add_metric()` 跟踪计算的精度指标：

```python
class LogisticEndpoint(keras.layers.Layer):
    def __init__(self, name=None):
        super(LogisticEndpoint, self).__init__(name=name)
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        self.accuracy_fn = keras.metrics.BinaryAccuracy()

    def call(self, targets, logits, sample_weights=None):
        # 计算训练时损失，并使用 `self.add_loss()` 添加到 layer
        loss = self.loss_fn(targets, logits, sample_weights)
        self.add_loss(loss)

        # 计算精度，并使用 `self.add_metric()` 添加到 layer
        acc = self.accuracy_fn(targets, logits, sample_weights)
        self.add_metric(acc, name="accuracy")

        # 返回预测张量
        return tf.nn.softmax(logits)
```

以这种方式记录的指标可以用 `layer.metrics` 查询：

```python
layer = LogisticEndpoint()

targets = tf.ones((2, 2))
logits = tf.ones((2, 2))
y = layer(targets, logits)

print("layer.metrics:", layer.metrics)
print("current accuracy value:", float(layer.metrics[0].result()))
```

```txt
layer.metrics: [<keras.metrics.metrics.BinaryAccuracy object at 0x0000018F8F2EAF10>]
current accuracy value: 1.0
```

与 `add_loss()` 一样，`fit()` 会自动记录这些指标：

```python
inputs = keras.Input(shape=(3,), name="inputs")
targets = keras.Input(shape=(10,), name="targets")
logits = keras.layers.Dense(10)(inputs)
predictions = LogisticEndpoint(name="predictions")(logits, targets)

model = keras.Model(inputs=[inputs, targets], outputs=predictions)
model.compile(optimizer="adam")

data = {
    "inputs": np.random.random((3, 3)),
    "targets": np.random.random((3, 10)),
}
model.fit(data)
```

```txt
1/1 [==============================] - 0s 387ms/step - loss: 0.8445 - binary_accuracy: 0.0000e+00
<keras.callbacks.History at 0x18f8ff26220>
```

## 8. 启用 layer 序列化

如果需要将自定义 layer 作为函数 API 的一部分序列化，可以实现 `get_config()` 方法：

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

    def get_config(self):
        return {"units": self.units}


# 使用 config 重新创建 layer
layer = Linear(64)
config = layer.get_config()
print(config)
new_layer = Linear.from_config(config)
```

```txt
{'units': 64}
```

基类 `Layer` 的 `__init__()` 方法包含一些关键字参数，如 `name` 和 `dtype`。最好在 `__init__()` 中将这些参数传递给父类：

```python
class Linear(keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(Linear, self).__init__(**kwargs)
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
        config = super(Linear, self).get_config()
        config.update({"units": self.units})
        return config


layer = Linear(64)
config = layer.get_config()
print(config)
new_layer = Linear.from_config(config)
```

```txt
{'name': 'linear_7', 'trainable': True, 'dtype': 'float32', 'units': 64}
```

如果从 config 反序列化 layer 需要更大的灵活性，则可以覆盖 `from_config()` 类方法。下面是 `from_config()` 的基本实现：

```python
def from_config(cls, config):
  return cls(**config)
```

## 9. call() 方法的 training 参数

某些 layeer，特别是 `BatchNormalization` 和 `Dropout` layer，在训练和推理过程具有不同的行为。对这类 layer，标准做法是在 `call()` 发方法中公开 `training` 参数：

```python
class CustomDropout(keras.layers.Layer):
    def __init__(self, rate, **kwargs):
        super(CustomDropout, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs
```

## 10. call() 方法的 mask 参数

`mask` 是 `call()` 支持的另一个参数。

在所有的 Keras RNN layer 中都可以找到该参数。`mask` 是一个布尔张量，输入中每个时间步对应一个布尔值，用于在处理时间序列数据时跳过输入中某些时间步。

对支持 `mask` 的 layer，Keras 会自动将上一层生成的 mask 以正确的参数传递给 `__call__()` 方法。生成 mask 的 layer 包括配置 `mask_zero=True` 的 `Embedding` layer 和 `Masking` layer。

## 11. Model 类

通常，使用 `Layer` 类定义内部计算；使用 `Model` 类定义外部模型，即需要训练的对象。

例如，在 ResNet50 模型中，包含多个继承 `Layer` 的 ResNet block，以及包含整个 ResNet50 网络的单个 `Model`。

`Model` 类与 `Layer` 具有相同的 API，具有以下差别：

- 包含内置的训练、评估和预测循环，即 `model.fit()`, `model.evaluate()` 和 `model.predict()`。
- 通过 `model.layers` 属性，公开其内部 layers
- 包含保存和序列化 API `save()`, `save_weights()`

实际上，`Layer` 类对应于文献中的 "layer"，如卷积层、循环层等，或者块（block），如 ResNet block, Inception block。

而 `Model` 类对应文献中的 "model"，即深度学习模型，或深度神经网络。

那么是使用 `Layer` 还是 `Model` 类呢？就看是否需要调用 `fit()`，是否需要 `save()`，如果是，就选择 `Model`；如果否，比如你定义的类是某个更大系统的一部分，或者你准备自己编写循环和保存代码，则使用 `Layer`。

例如，以上面的 mini-resnet 为例，使用它构建 `Model`，这样就可以使用 `fit()` 训练模型，使用 `save_weights()` 保存权重：

```python
class ResNet(tf.keras.Model):

    def __init__(self, num_classes=1000):
        super(ResNet, self).__init__()
        self.block_1 = ResNetBlock()
        self.block_2 = ResNetBlock()
        self.global_pool = layers.GlobalAveragePooling2D()
        self.classifier = Dense(num_classes)

    def call(self, inputs):
        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.global_pool(x)
        return self.classifier(x)


resnet = ResNet()
dataset = ...
resnet.fit(dataset, epochs=10)
resnet.save(filepath)
```

## 12. 完整示例

对上面的内容进行总结：

- `Layer` 封装了状态（在 `__init__()` 或 `build()` 中创建）和计算（在 `call()` 中定义）
- layer 可以递归嵌套，以创建更大的计算 block
- layer 可以通过 `add_loss()` 和 `add_metric()` 记录损失值（通常是正则化损失）和指标
- `Model` 为外层容器，是需要训练的对象。`Model` 和 `Layer` 类似，但是增加了训练和序列化工具。

现在我们将所有这些组合在一起，创建一个端到端的示例，实现一个变分自动编码器（Variational AutoEncoder, VAE），并在 MNIST 数据集上训练。

VAE 继承 `Model` 类，由 `Layer` 的子类嵌套组成。它包含正则化损失（KL divergence）。

```python
from tensorflow.keras import layers


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim=32, intermediate_dim=64, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, original_dim, intermediate_dim=64, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
        self.dense_output = layers.Dense(original_dim, activation="sigmoid")

    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)


class VariationalAutoEncoder(keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(
            self,
            original_dim,
            intermediate_dim=64,
            latent_dim=32,
            name="autoencoder",
            **kwargs
    ):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed
```

在 MNIST 上编写一个简单的训练循环：

```python
original_dim = 784
vae = VariationalAutoEncoder(original_dim, 64, 32)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
mse_loss_fn = tf.keras.losses.MeanSquaredError()

loss_metric = tf.keras.metrics.Mean()

(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype("float32") / 255

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

epochs = 2

# Iterate over epochs.
for epoch in range(epochs):
    print("Start of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step, x_batch_train in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            reconstructed = vae(x_batch_train)
            # Compute reconstruction loss
            loss = mse_loss_fn(x_batch_train, reconstructed)
            loss += sum(vae.losses)  # Add KLD regularization loss

        grads = tape.gradient(loss, vae.trainable_weights)
        optimizer.apply_gradients(zip(grads, vae.trainable_weights))

        loss_metric(loss)

        if step % 100 == 0:
            print("step %d: mean loss = %.4f" % (step, loss_metric.result()))
```

```txt
Start of epoch 0
step 0: mean loss = 0.3316
step 100: mean loss = 0.1255
step 200: mean loss = 0.0991
step 300: mean loss = 0.0891
step 400: mean loss = 0.0842
step 500: mean loss = 0.0809
step 600: mean loss = 0.0787
step 700: mean loss = 0.0771
step 800: mean loss = 0.0760
step 900: mean loss = 0.0750
Start of epoch 1
step 0: mean loss = 0.0747
step 100: mean loss = 0.0740
step 200: mean loss = 0.0735
step 300: mean loss = 0.0730
step 400: mean loss = 0.0727
step 500: mean loss = 0.0723
step 600: mean loss = 0.0720
step 700: mean loss = 0.0717
step 800: mean loss = 0.0715
step 900: mean loss = 0.0712
```

由于 VAE 是 `Model` 的子类，因此它内置有训练循环。所以也可以按如下方法训练：

```python
vae = VariationalAutoEncoder(784, 64, 32)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
vae.fit(x_train, x_train, epochs=2, batch_size=64)
```

```txt
Epoch 1/2
938/938 [==============================] - 5s 5ms/step - loss: 0.0750
Epoch 2/2
938/938 [==============================] - 5s 5ms/step - loss: 0.0676
<keras.callbacks.History at 0x18f8f7f5790>
```

## 13. 函数 API

这个示例是面向对象的代码风格，也可以使用函数 API 构建模型。最重要的是，这两种风格的 API 不是互斥的，也可以混合搭配使用。

例如，下面的函数 API 示例重用上面定义的 `Sampling` layer：

```python
original_dim = 784
intermediate_dim = 64
latent_dim = 32

# Define encoder model.
original_inputs = tf.keras.Input(shape=(original_dim,), name="encoder_input")
x = layers.Dense(intermediate_dim, activation="relu")(original_inputs)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()((z_mean, z_log_var))
encoder = tf.keras.Model(inputs=original_inputs, outputs=z, name="encoder")

# Define decoder model.
latent_inputs = tf.keras.Input(shape=(latent_dim,), name="z_sampling")
x = layers.Dense(intermediate_dim, activation="relu")(latent_inputs)
outputs = layers.Dense(original_dim, activation="sigmoid")(x)
decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name="decoder")

# Define VAE model.
outputs = decoder(z)
vae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name="vae")

# Add KL divergence regularization loss.
kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
vae.add_loss(kl_loss)

# Train.
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
vae.fit(x_train, x_train, epochs=3, batch_size=64)
```

```txt
Epoch 1/3
938/938 [==============================] - 5s 5ms/step - loss: 0.0746
Epoch 2/3
938/938 [==============================] - 4s 5ms/step - loss: 0.0676
Epoch 3/3
938/938 [==============================] - 4s 5ms/step - loss: 0.0676
<keras.callbacks.History at 0x190485e1670>
```

## 14. 参考

- https://www.tensorflow.org/guide/keras/custom_layers_and_models
