# TensorFlow 2 快速入门（专家）

- [TensorFlow 2 快速入门（专家）](#tensorflow-2-快速入门专家)
  - [参考](#参考)

2021-12-30, 14:04
@author Jiawei Mao
***

下载并安装 TensorFlow 2，导入 TensorFlow：

```python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
```

```sh
TensorFlow version: 2.7.0
```

载入并准备 MNIST 数据集：

```python
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")
```

使用 `tf.data` 执行 batch （分批） 和 shuffle（打乱）数据集操作：

```python
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
```

使用 `tf.keras` 的扩展父类 API 构建模型：

```python
class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10)

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

# Create an instance of the model
model = MyModel()
```

选择合适的优化器和损失函数：

```python
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()
```

选择衡量模型的损失和准确性的指标（metric）。metric 值在每个 epoch 完成后输出一次：

```python
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
```

使用 `tf.GradientTap` 训练模型：

```python
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)
```

测试模型：

```python
@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)
```

```python
EPOCHS = 5

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
    f'Test Loss: {test_loss.result()}, '
    f'Test Accuracy: {test_accuracy.result() * 100}'
  )
```

```sh
Epoch 1, Loss: 0.13439151644706726, Accuracy: 95.98833465576172, Test Loss: 0.06330505013465881, Test Accuracy: 98.02999877929688
Epoch 2, Loss: 0.04413021355867386, Accuracy: 98.63166809082031, Test Loss: 0.0737706869840622, Test Accuracy: 97.57999420166016
Epoch 3, Loss: 0.024562617763876915, Accuracy: 99.1866683959961, Test Loss: 0.04913036897778511, Test Accuracy: 98.5
Epoch 4, Loss: 0.014813972637057304, Accuracy: 99.51499938964844, Test Loss: 0.06434403359889984, Test Accuracy: 98.1199951171875
Epoch 5, Loss: 0.010033201426267624, Accuracy: 99.66999816894531, Test Loss: 0.07429419457912445, Test Accuracy: 98.15999603271484
```

一个图像分类器到这儿就完成了，在该数据集上的准确度达到 ~98%。

## 参考

- https://www.tensorflow.org/tutorials/quickstart/advanced
