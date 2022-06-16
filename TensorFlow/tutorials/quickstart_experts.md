# TensorFlow 2 快速入门（专家）

Last updated: 2022-06-16, 15:26
@author Jiawei Mao
****

下载并安装 TensorFlow 2，导入 TensorFlow：

```python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
```

```txt
TensorFlow version: 2.9.1
```

加载并准备 MNIST 数据集：

```python
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 添加通道维度
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")
```

使用 `tf.data` 执行 batch（批量） 和 shuffle（打乱）数据集操作：

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


# 创建模型实例
model = MyModel()
```

设置优化器和损失函数：

```python
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()
```

选择衡量模型的损失和准确性的指标（metric）。metric 值在每个 epoch 完成后输出：

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
        # 只有在包含训练和推理时行为不同的 layer 时（如 Dropout），
        # 才需要设置 training=True
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
    # 只有包含训练和推理时行为不同的 layer 时（如 Dropout），
    # 才需要设置 training=False
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)
```

```python
EPOCHS = 5

for epoch in range(EPOCHS):
    # 在 epoch 开始前重置 metrics
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

```txt
Epoch 1, Loss: 0.13300782442092896, Accuracy: 95.98666381835938, Test Loss: 0.06714330613613129, Test Accuracy: 97.65999603271484
Epoch 2, Loss: 0.04121926426887512, Accuracy: 98.70832824707031, Test Loss: 0.05591356009244919, Test Accuracy: 98.15999603271484
Epoch 3, Loss: 0.021372780203819275, Accuracy: 99.29500579833984, Test Loss: 0.05870741233229637, Test Accuracy: 98.1199951171875
Epoch 4, Loss: 0.013534549623727798, Accuracy: 99.55500030517578, Test Loss: 0.053924888372421265, Test Accuracy: 98.3499984741211
Epoch 5, Loss: 0.00904169213026762, Accuracy: 99.68666076660156, Test Loss: 0.06500118225812912, Test Accuracy: 98.30999755859375
```

一个图像分类器到这儿就完成了，在该数据集上的准确度约为 98%。

## 参考

- https://www.tensorflow.org/tutorials/quickstart/advanced
