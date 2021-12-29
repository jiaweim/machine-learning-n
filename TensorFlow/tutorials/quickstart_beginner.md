# TensorFlow 2 快速入门（初学者）

- [TensorFlow 2 快速入门（初学者）](#tensorflow-2-快速入门初学者)
  - [简介](#简介)
  - [导入 TensorFlow](#导入-tensorflow)
  - [载入数据集](#载入数据集)
  - [构建机器学习模型](#构建机器学习模型)
  - [训练和评估模型](#训练和评估模型)
  - [参考](#参考)

2021-12-28, 15:22
@author Jiawei Mao
***

## 简介

这篇入门教程，介绍使用 Keras 执行如下操作：

1. 载入预先构建的数据集；
2. 建立一个图像分类的神经网络机器学习模型；
3. 训练神经网络；
4. 评估模型的准确性。

## 导入 TensorFlow

首先导入 TensorFlow：

```python
import tensorflow as tf

print("TensorFlow version: ", tf.__version__)
```

```sh
TensorFlow version:  2.7.0
```

## 载入数据集

载入 MNIST 数据集，将数据从整数转换为浮点数：

```python
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```

## 构建机器学习模型

通过叠加层建立 `tf.keras.Sequential` 模型：

```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
```

对每个示例，模型都返回一个 logit 向量，每个类别对应一个值。例如：

```python
predictions = model(x_train[:1]).numpy()
predictions
```

```sh
array([[ 0.27800095, -0.18243524, -0.32130682, -0.0138693 , -0.05083328,
         0.22902712, -0.9523146 ,  0.23277804, -0.18671265, -0.11169323]],
      dtype=float32)
```

`tf.nn.softmax` 函数可以将 logit 值转换为概率值：

```python
tf.nn.softmax(predictions).numpy()
```

```sh
array([[0.13980936, 0.08822086, 0.07678213, 0.10441878, 0.1006295 ,
        0.13312732, 0.04085234, 0.13362761, 0.08784432, 0.09468783]],
      dtype=float32)
```

> 注意：可以将 `tf.nn.softmax` 函数作为神经网络最后一层的激活函数。这可以使模型直接输出可解释的值，但不鼓励使用这种方法，因为 softmax 输出无法为所有模型提供精确且数值稳定的损失计算。

使用 `losses.SparseCategoricalCrossentropy` 定义训练所需损失函数：

```python
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

该损失值等于真实类别概率的负对数，如果模型预测正确，损失值为 0。

未经训练的模型给出的概率接近随机（每个类的概率为 1/10），因此初始损失应该接近于 $-tf.math.log(1/10) ~= 2.3$。

```python
loss_fn(y_train[:1], predictions).numpy()
```

```sh
2.2223506
```

在训练之前，使用 Keras 的 `Model.compile` 配置和编译模型。将 `optimizer` 设置为 `adam`，将 `loss` 设置为前面定义的 `loss_fn`，将评估模型的度量 `metrics` 参数设置为 `accuracy`。

```python
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
```

## 训练和评估模型

使用 `Model.fit` 调整模型参数，最小化损失：

```python
model.fit(x_train, y_train, epochs=5)
```

```sh
Epoch 1/5
1875/1875 [==============================] - 5s 2ms/step - loss: 0.2992 - accuracy: 0.9149
Epoch 2/5
1875/1875 [==============================] - 4s 2ms/step - loss: 0.1454 - accuracy: 0.9579
Epoch 3/5
1875/1875 [==============================] - 4s 2ms/step - loss: 0.1091 - accuracy: 0.9668
Epoch 4/5
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0895 - accuracy: 0.9721
Epoch 5/5
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0756 - accuracy: 0.9769
<keras.callbacks.History at 0x1a831c5fa00>
```

`Model.evaluate` 方法用于检查模型性能，通常对验证集或测试集数据。

```python
model.evaluate(x_test,  y_test, verbose=2)
```

```sh
313/313 - 1s - loss: 0.0717 - accuracy: 0.9775 - 598ms/epoch - 2ms/step
[0.0716572180390358, 0.9775000214576721]
```

分类器在这个数据集上的准确率接近 98%。

如果你希望模型返回一个概率，可以把训练好的模型包装起来，额外加一个 softmax：

```python
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
```

```python
probability_model(x_test[:5])
```

```sh
<tf.Tensor: shape=(5, 10), dtype=float32, numpy=
array([[2.1194946e-07, 8.5832559e-09, 2.2455561e-05, 1.7866098e-04,
        7.7193300e-11, 8.9379677e-09, 3.6577387e-13, 9.9976498e-01,
        5.7338508e-07, 3.3120705e-05],
       [9.8032831e-07, 5.8613041e-05, 9.9984479e-01, 8.1006190e-05,
        5.5317058e-15, 2.8216211e-06, 5.3067456e-08, 1.3904465e-12,
        1.1677374e-05, 2.6266595e-11],
       [9.9731892e-08, 9.9956959e-01, 5.6959343e-05, 7.9618903e-06,
        6.0599523e-06, 1.4311652e-05, 3.9742954e-05, 1.0153104e-04,
        2.0328345e-04, 4.0050264e-07],
       [9.9984145e-01, 7.8144707e-10, 1.1152010e-05, 3.0016800e-08,
        1.2465817e-06, 5.3614735e-06, 7.8149124e-06, 1.4401690e-05,
        7.1907046e-08, 1.1838077e-04],
       [1.4015175e-06, 3.6343459e-09, 6.8545132e-06, 1.5601460e-09,
        9.9748719e-01, 6.1192473e-08, 9.9746330e-07, 3.1162304e-04,
        5.5570689e-07, 2.1913524e-03]], dtype=float32)>
```

恭喜，到这里你已经使用 keras API 训练好了一个机器学习模型。

## 参考

- https://www.tensorflow.org/tutorials/quickstart/beginner
