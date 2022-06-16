# TensorBoard 入门

- [TensorBoard 入门](#tensorboard-入门)
  - [简介](#简介)
  - [在 Keras Model.fit 中使用 TensorBoard](#在-keras-modelfit-中使用-tensorboard)
  - [在其它方法中使用 TensorBoard](#在其它方法中使用-tensorboard)
  - [参考](#参考)

@author Jiawei Mao
***

## 简介

在机器学习中，要改善某个指标，首先需要对其进行计算。TensorBoard 提供了机器学习中指标值的计算和可视化功能。它可以追踪实验指标（如损失值、准确性）、可视化模型图、将嵌入映射到较低维度等。

下面演示如何使用 TensorBoard。

- 加载 TensorBoard notebook 扩展

```python
%load_ext tensorboard
```

```python
import tensorflow as tf
import datetime
```

清除以前运行留下的日志：

```powershell
rm -rf ./logs/
```

以 MNIST 数据集为例，对数据进行归一化，并创建一个简单的 Keras 模型将图片分为 10 类：

```python
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
```

```txt
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11490434/11490434 [==============================] - 1s 0us/step
```

## 在 Keras Model.fit 中使用 TensorBoard

使用 Keras 的 `Model.fit()` 训练模型时，添加 `tf.keras.callbacks.TensorBoard` callback 以确保创建和存储日志。另外，设置 `histogram_freq=1` 启用直方图计算（默认关闭）。

将日志放在带时间戳的子目录，便于选择而不同的训练日志：

```python
model = create_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x=x_train,
          y=y_train,
          epochs=5,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard_callback])
```

```txt
Epoch 1/5
1875/1875 [==============================] - 6s 3ms/step - loss: 0.2225 - accuracy: 0.9343 - val_loss: 0.1184 - val_accuracy: 0.9633
Epoch 2/5
1875/1875 [==============================] - 6s 3ms/step - loss: 0.0965 - accuracy: 0.9708 - val_loss: 0.0900 - val_accuracy: 0.9725
Epoch 3/5
1875/1875 [==============================] - 6s 3ms/step - loss: 0.0679 - accuracy: 0.9784 - val_loss: 0.0751 - val_accuracy: 0.9770
Epoch 4/5
1875/1875 [==============================] - 6s 3ms/step - loss: 0.0542 - accuracy: 0.9829 - val_loss: 0.0740 - val_accuracy: 0.9778
Epoch 5/5
1875/1875 [==============================] - 6s 3ms/step - loss: 0.0437 - accuracy: 0.9858 - val_loss: 0.0746 - val_accuracy: 0.9784
```

通过命令行或 botebook 启动 TensorBoard。这两个接口通常一样。在 notebook 中，使用 `%tensorboard` magic，在命令行，运行不带 "%" 的命令。

```python
%tensorboard --logdir logs/fit
```

![](images/2022-06-16-13-30-14.png)

TensorBoard 主要包括：

- **SCALARS** 面板展示损失值等指标随 epoch 的变化。使用该面板还可以跟踪训练速度、学习速率以及其它标量值。
- **GRAPHS** 面板用于可视化模型。对该示例，显示 Keras 模型图，有助于确保模型的正确。
- **Distributions** 和 **Histograms** 面板显示张量随时间的分布。这有助于可视化权重和偏差，查看它们是否按照预期的方式在变化。

当记录其它类型的数据时，会自动启用与其对应的 TensorBoard 插件。例如，Keras TensorBoard callback 可以记录图像和嵌入。点击右下角的 "inactive" 下拉框可以看到其它插件。

## 在其它方法中使用 TensorBoard

当使用 `tf.GradientTape()` 等方法训练时，使用 `tf.summary` 记录所需信息。

还是使用 MNIST 数据集，但将其转换为 `tf.data.Dataset` 以利用批处理功能：

```python
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_dataset = train_dataset.shuffle(60000).batch(64)
test_dataset = test_dataset.batch(64)
```



## 参考

- https://www.tensorflow.org/tensorboard/get_started
